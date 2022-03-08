#imports

import os
import nibabel as ni
import numpy as np
import pytorch_lightning as pl
import argparse
from glob import glob

import torch
from CNNs.unet import UNet
from Utils.transforms import My_transforms

import Utils.view as vi
import Utils.Metrics as mt



def load_nii_file(file_path):
    data = ni.load(file_path)
    volume = np.nan_to_num(data.get_data().squeeze())
    return volume

def load_files(file_paths, d_type=None):
    images = []
    for path in file_paths:
        if d_type == None:
            images.append(load_nii_file(path))
        else: 
            images.append(load_nii_file(path).astype(d_type))
    return images


def to_onehot(matrix, labels=[], single_foregound_lable=True, background_channel=True, onehot_type=np.dtype(np.float32)):
    matrix = np.around(matrix)
    if len(labels) == 0:
        labels = np.unique(matrix) 
        labels = labels[1::]
    
    mask = np.zeros(matrix.shape, dtype=onehot_type)
    for i, label in enumerate(labels):
        mask += ((matrix == label) * (i+1))
   
    if single_foregound_lable:
        mask = (mask > 0)
        labels = [1]
        
    labels_len = len(labels)        
        
    onehot = np.zeros((labels_len+1,) + matrix.shape, dtype=onehot_type) 
    for i in range(mask.max()+1):
        onehot[i] = (mask == i)  
        
    if background_channel == False:
        onehot = onehot[1::] 
        
       
    return mask, onehot, labels

class Segmentor(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.save_hyperparameters(hparams)        

        if "unet" in self.hparams.cnn_architecture:
            architecture = UNet(nin_channels=self.hparams.n_inchannels, 
                                nout_channels=self.hparams.n_outchannels, 
                                init_features=self.hparams.init_features)
        elif self.hparams.cnn_architecture == "coedet":
            architecture = CoEDET(nin=self.hparams.n_inchannels, nout=self.hparams.n_outchannels, 
                                  apply_sigmoid=self.hparams.apply_sigmoid)
        else:
            raise ValueError(f"Unsupported cnn_architecture {self.hparams.cnn_architecture}")

#         self.model = architecture(self.hparams)
        self.model = architecture
    
        
        ttransform_scale=None
        ttransform_angle=None
        ttransform_flip_prob=None
        ttransform_sigma=None
        ttransform_ens_treshold=None
        if "taug_scale" in self.hparams:
            ttransform_scale = self.hparams.taug_scale
        if "taug_angle" in self.hparams:
            ttransform_angle = self.hparams.taug_angle
        if "taug_flip_prob" in self.hparams:
            ttransform_flip_prob = self.hparams.taug_flip_prob
        if "taug_sigma" in self.hparams:
            ttransform_sigma = self.hparams.taug_sigma
        if "taug_ens_treshold" in self.hparams:
            ttransform_ens_treshold = self.hparams.aug_ens_treshold
        self.train_transforms = My_transforms(scale=ttransform_scale,
                                         angle=ttransform_angle,
                                         flip_prob=ttransform_flip_prob,
                                         sigma=ttransform_sigma,
                                         ens_treshold=ttransform_ens_treshold
                                        )
        vtransform_scale=None
        vtransform_angle=None
        vtransform_flip_prob=None
        vtransform_sigma=None
        vtransform_ens_treshold=None
        if "vaug_scale" in self.hparams:
            ttransform_scale = self.hparams.vaug_scale
        if "vaug_angle" in self.hparams:
            ttransform_angle = self.hparams.vaug_angle
        if "vaug_flip_prob" in self.hparams:
            ttransform_flip_prob = self.hparams.vaug_flip_prob
        if "vaug_sigma" in self.hparams:
            ttransform_sigma = self.hparams.vaug_sigma
        if "vaug_ens_treshold" in self.hparams:
            ttransform_ens_treshold = self.hparams.vaug_ens_treshold
        self.val_transforms = My_transforms(scale=vtransform_scale,
                                         angle=vtransform_angle,
                                         flip_prob=vtransform_flip_prob,
                                         sigma=vtransform_sigma,
                                         ens_treshold=vtransform_ens_treshold
                                        )  

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        loss = None

        x, y = train_batch
        logits = self.forward(x)
#         print('x.shape = ', x.shape)
#         print('y.shape = ', y.shape)
#         print('logits.shape = ', logits.shape)

        loss = CombinedLoss(logits, y, 
                            self.hparams.train_loss_funcs, 
                            self.hparams.lossweighs,
                            func_weights=self.hparams.func_weights)

        if self.hparams.train_metric == 'DiceMetric_weighs':
            train_metric = DiceMetric_weighs(y_pred=logits, y_true=y,
                                             weights=self.hparams.train_metricweighs, treshold=0.5)
        else:
            raise ValueError(f"Unsupported metric {self.hparams.train_metric}")

        self.log("loss", loss, on_epoch=True, on_step=True)
        self.log("train_metric", train_metric, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        logits = None

        x, y = val_batch
        logits = self.forward(x)
#         loss = self.lossfunc(logits, y)

        loss = CombinedLoss(logits, y, 
                            self.hparams.val_loss_funcs, 
                            self.hparams.lossweighs,
                            func_weights=self.hparams.func_weights)
    
        if self.hparams.val_metric == 'DiceMetric_weighs':
            val_metric = DiceMetric_weighs(y_pred=logits, y_true=y,
                                             weights=self.hparams.val_metricweighs, treshold=0.5)
        else:
            raise ValueError(f"Unsupported metric {self.hparams.val_metric}")

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_metric", val_metric, on_epoch=True, on_step=False, prog_bar=True)
        self.log("learning_rate_test", self.optimizer.param_groups[0]['lr'], on_epoch=True, on_step=False, prog_bar=False)

    
    def get_optimizer_by_name(self, name, lr):
        if name == "Adam":
            return Adam(self.model.parameters(), lr=lr)
        elif name == "SGD":
            return SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
            

    def configure_optimizers(self):
        optimizer = self.get_optimizer_by_name(self.hparams.opt_name, 
                                               self.hparams.lr)

        if self.hparams.lr_decay_policy == 'step':
            scheduler = StepLR(optimizer, self.hparams.scheduling_patience_lrepochs, self.hparams.lr_decay_factor, verbose=True)
            #print('STEP - scheduling_patience_lrepochs = ', self.hparams.scheduling_patience_lrepochs, ' lr_decay_factor = ', self.hparams.lr_decay_factor)
        elif self.hparams.lr_decay_policy == 'plateau':
            #print('PLATEAU - scheduling_patience_lrepochs = ', self.hparams.scheduling_patience_lrepochs, ' lr_decay_factor = ', self.hparams.lr_decay_factor)

            self.optimizer = optimizer
            lr_scheduler =  {
                           'scheduler': ReduceLROnPlateau(optimizer),
                           'mode': self.hparams.lr_decay_mode,
                           'factor': self.hparams.lr_decay_factor,
                           'patience': self.hparams.scheduling_patience_lrepochs,
                           'threshold': self.hparams.learning_threshold,
                           'threshold_mode': self.hparams.lr_decay_threshold_mode,
                           'cooldown': 0,
#                            'min_lr': self.hparams.lr,
                           'min_lr': self.hparams.min_lr,
                           'eps': self.hparams.eps,
                           'monitor': self.hparams.monitor,
                           'verbose': True
                           }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        
        else:
            raise ValueError(f"Unsupported lr_decay_policy {self.hparams.lr_decay_policy}")
            

        return [optimizer], [scheduler] 
    

def find_checkpoint(pre_trained_ckpt_folder,view,input_channels):
    ckpt_candidates = glob(f"{pre_trained_ckpt_folder}/*{view}_{''.join(input_channels)}_*.ckpt")
    #print(f"{pre_trained_ckpt_folder}/*{view}_{''.join(input_channels)}_*.ckpt")
    #print('ckpt_candidates = ', ckpt_candidates)
    assert len(ckpt_candidates) == 1, "More than one checkpoint elegible, leave only one checkpoint for each view and data combination."
#     print(f"Found following {view} checkpoint: {ckpt_candidates}.")
    #print(view)
    return ckpt_candidates[0]


def ch2sufix(input_channels):
    img_paths = []
    for in_ch in input_channels:
        if in_ch == 'evalue1':
            img_paths.append('diffusion/evalue1.nii.gz')
        if in_ch == 'FA':
            img_paths.append('diffusion/FA.nii.gz')
        if in_ch == 'RD':
            img_paths.append('diffusion/RD.nii.gz')
        if in_ch == 'MD':
            img_paths.append('diffusion/MD.nii.gz')
        if in_ch == 'T1':
            img_paths.append('T1w_acpc_dc_restore_1.25.nii.gz')
    return img_paths
    
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Thalamus segmentation on diffusion and T1 data.')
	parser.add_argument('--in_folder', '-in', required=True, help='Folder path with subjects')
	parser.add_argument('--model_path', '-mp', required=True, help='Folder path with CNN models')
	parser.add_argument('--channel_1', '-ch1', required=True, help='CNN input channel 1: evalue1, FA, RD, MD, T1')
	parser.add_argument('--channel_2', '-ch2', required=False, help='CNN input channel 2: evalue1, FA, RD, MD, T1')
	parser.add_argument('--channel_3', '-ch3', required=False, help='CNN input channel 3: evalue1, FA, RD, MD, T1')
	parser.add_argument('--channel_4', '-ch4', required=False, help='CNN input channel 4: evalue1, FA, RD, MD, T1')
	parser.add_argument('--channel_5', '-ch5', required=False, help='CNN input channel 5: evalue1, FA, RD, MD, T1')
	parser.add_argument('--model_orientation', '-mo', required=False, default='3view', help='Slice orientation of the model: axial, coronal, sagittal, 3view (default) ')
	parser.add_argument('--out_folder', '-out', required=True, help='Folder path for output segmentation')
	parser.add_argument('--verbose', '-v', required=False, help='print verbose', action="store_true")
	args = parser.parse_args()

    	   
    
    
	# Paths
	pre_trained_ckpt_folder = args.model_path #"fine_tuning_unet_single_label_freeze/"
	# dataset_folder = './Data/HCP_processed_data/'
	dataset_folder = args.in_folder #'../../HCP_processed_data/'
	subject_list = 'testing_subjects.txt'
	prediction_folder = args.out_folder #f"./Predictions/{pre_trained_ckpt_folder}"

	if args.model_orientation == '3view':
		# Experiment definition
		Slice_views = ['axial', 'coronal', 'sagittal']
	else: 
		Slice_views = [args.model_orientation]


	input_channels = []
	if args.channel_1: input_channels.append(args.channel_1)
	if args.channel_2: input_channels.append(args.channel_2)
	if args.channel_3: input_channels.append(args.channel_3)
	if args.channel_4: input_channels.append(args.channel_4)
	if args.channel_5: input_channels.append(args.channel_5)
	
	# input_channels=['evalue1', 'FA', 'RD', 'MD', 'T1']
	# input_channels=['evalue1', 'FA', 'RD', 'MD']
	# input_channels=['T1']
	# input_channels=['FA', 'RD']
	# input_channels=['FA', 'T1']
	# input_channels=['FA', 'MD']
	# input_channels=['FA', 'evalue1']
	# input_channels=['RD', 'MD']
	# input_channels=['FA']
	# input_channels=['MD']
	# input_channels=['RD']
	# input_channels=['evalue1']
	# input_channels=['MD', 'T1']
	# input_channels=['evalue1', 'T1']
	

 


	percentil_filt = 99.98
	normalize_volumes = [0,1]
	prediction_threshold = 0.5
	input_d_type='float32'
	save_prediction = True



	dest_folder = f"{prediction_folder}{'-'.join(Slice_views)}_{''.join(input_channels)}/"
	os.makedirs(dest_folder, exist_ok=True)
	mask_free_sufix = 'FreeSurfer/aparc+aseg_1.25_nearest.nii.gz'
	subjects = os.listdir(dataset_folder)
	
	if args.verbose:
		print('pre_trained_ckpt_folder', pre_trained_ckpt_folder)
		print('dataset_folder', dataset_folder)
		print('prediction_folder', prediction_folder)
		print('model_orientation', args.model_orientation)
		print('Slice_views', Slice_views)
		print('input_channels', input_channels)
		print('dest_folder', dest_folder)
		print('listdir subjects', subjects)
	
	#subjects = [line.strip() for line in open(subject_list)]

	MASKS = []
	STAPLE = []
	FREE = []
	FSL = []
	QUI = []
	MAN = []
	PREDICTIONS = []
	PREDICTIONS_fullsize = []

	for subject in subjects:
		if args.verbose:
			print('subject = ', subject)
		
		img_paths = ch2sufix(input_channels)
		images = load_files([dataset_folder + subject + '/' +  s for s in img_paths])

		if percentil_filt > 0:
			for i in range(len(images)):
				images[i][images[i] > np.percentile(images[i], percentil_filt)] = np.percentile(images[i], percentil_filt)
	    
		if len(normalize_volumes) == 2:
			for i in range(len(images)):
				images[i] = images[i] * ((normalize_volumes[1]-normalize_volumes[0])/(images[i].max()-images[i].min()))
				images[i] = images[i] - images[i].min() + normalize_volumes[0]          
	   
		img_crop = np.array(images)[:, :144, 15:159, :144]
	    
		PREDS = []
		for Slice_view in Slice_views:
			if args.verbose:
				print('Slice_view = ', Slice_view)
		
			# reorient images
			if Slice_view == 'axial':
		    		img_crop_reoriented = np.transpose(img_crop, (3, 0, 1, 2))
			elif Slice_view == 'coronal':
		    		img_crop_reoriented = np.transpose(img_crop, (2, 0, 1, 3))
			elif Slice_view == 'sagittal':
		    		img_crop_reoriented = np.transpose(img_crop, (1, 0, 2, 3))
		    
			model_path = find_checkpoint(pre_trained_ckpt_folder,Slice_view,input_channels)

			trained_model = Segmentor.load_from_checkpoint(model_path).eval()

			with torch.no_grad():
		    		preds = trained_model(torch.tensor(img_crop_reoriented)).cpu().numpy()

			# reorient images
			if Slice_view == 'axial':
		    		preds = np.transpose(preds, (1, 2, 3, 0))
			elif Slice_view == 'coronal':
		    		preds = np.transpose(preds, (1, 2, 0, 3))
			elif Slice_view == 'sagittal':
		    		preds = np.transpose(preds, (1, 0, 2, 3))

			PREDS.append(preds) #prediction for each slice
	    
		prediction = np.zeros(preds.shape)
		for pred in PREDS:
			prediction = prediction + pred/len(PREDS)
	    
		if save_prediction:
			FREE_file = ni.load(dataset_folder + subject + '/' + mask_free_sufix)
			FREE_data = FREE_file.get_data()  
			            
			PREDICTION_fullsize = np.zeros(images[0].shape)
			PREDICTION_fullsize[:144, 15:159, :144] = (prediction[1] >= prediction_threshold)  #save only the thalamus channel
			PREDICTIONS_fullsize.append(PREDICTION_fullsize)
			prediction_file = ni.Nifti1Image(PREDICTION_fullsize.astype(FREE_data.dtype), affine=FREE_file.affine, header=FREE_file.header)
			ni.save(prediction_file, dest_folder + subject + '.nii.gz')
		
		PREDICTIONS.append(np.asarray(prediction))
    
    
# python3 inference.py -in ../../subs2inference/ -mp ./checkpoints/fine_tuning_unet_single_label_freeze/ -ch1=evalue1 -ch2 T1 -out ./Predictions/fine_tuning_unet_single_label_freeze/ -v 
# python3 inference.py -in ../../subs2inference/ -mp ./checkpoints/fine_tuning_unet_single_label_freeze/ -ch1=evalue1 -ch2 T1 -mo axial -out ./Predictions/fine_tuning_unet_single_label_freeze/ -v
# python3 inference.py -in ../../subs2inference/ -mp ./checkpoints/unet_single_label_attention/ -ch1 T1 -out ./Predictions/unet_single_label_attention/ -v #nao deu certo
# python3 inference.py -in ../../subs2inference/ -mp ./checkpoints/fine_tuning_unet_single_label_freeze_old/ -ch1 T1 -out ./Predictions/fine_tuning_unet_single_label_freeze_old/ -v 
# python3 inference.py -in ../../subs2inference/ -mp ./checkpoints/fine_tuning_unet_single_label_freeze_old/ -ch1 FA -ch2 RD -out ./Predictions/fine_tuning_unet_single_label_freeze_old/ -v 
# python3 inference.py -in /Data/subs2inference/ -mp ./checkpoints/fine_tuning_unet_single_label_freeze_old/ -ch1 FA -ch2 RD -out ./Predictions/fine_tuning_unet_single_label_freeze_old/ -v 

