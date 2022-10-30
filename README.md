# thalamus_benchmark_diffusion

# Introduction
This work is a reproducibility package for the Thalamus Segmentation Benchmark https://www.ccdataset.com/thalamus-benchmark.
Our paper describing everything in details is in the submission process and a citation will be made available in the future.

# Data Acquisition

To acquire de processed data, follow the steps on https://www.ccdataset.com/thalamus-benchmark.

Extract it following the existing organization on the code/Data folder. 
After populating the code/Data folder, you should be able to run every necessary step of the pipeline. 

# Benchmark Submission and Leaderboard

The volumes for testing your thalamus segmentation model and submission instructions are provided in https://www.ccdataset.com/thalamus-benchmark.

The leaderboard is currently as follows: 

To be announced.

# Docker Environment

Install Docker and nVidia GPU support for your Linux distribution following the tutorial at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker.

Pull our environment image with the following command:

    docker pull dscarmo/thalamus_env

Pulling the image is our recommended approach. However, if you want to build the Docker image locally from scratch run the following command:

    sh build.sh 

Finally, make sure port 8888 is not in use (by a Jupyter Notebook instance for example) and run the environment with:

    python run.py

A notebook link will be displayed on your terminal. Open it on your browser to access the notebooks in the code folder.

# Requirements


If you don't want to use a Docker Environment, you have to setup a PyTroch installation with GPU support. Install the 1.8 version following the guide on https://pytorch.org.

Additional libraries required are as follows (as in the requirements.txt file):

notebook\
numpy==1.19.2\
nibabel==3.2.1\
pytorch_lightning==1.3.8\
pandas==1.1.4\
seaborn==0.11.2\
SimpleITK==2.0.2\
scikit-image==0.17.2\
dipy==1.4.1\
connected-components-3d==2.0.0\

This work was only tested in Ubuntu 20.04. We don't guarantee support outside of using the Docker environment.

# Code Usage

The notebooks represent each processing stage for the pipeline of this work. The notebooks are enumerated according to the run order.

1 - Patche_creation.ipynb

This code creates the patches to be used during the training phase. It creates patches for all input channels as well as for all segmentation masks.

2 - Training.ipynb

Used to train the CNN models. Parameters such as the combination of input channels are allowed to be changed.

3 - Patche_creation_finetuning.ipynb

Equivalent to 1, but specifically for the fine tuning subset.

4 - Finetuning.ipynb

Used to finetune the models trained in notebook number 2.

5 - Save inference (test set _20 subjects).ipynb

This notebook uses any trained model to make inferences and save the prediction in .nii format. It could be done on the test subset or any other subgroup.

6 - Evaluate_prediction.ipynb

Notebook used to compute the metrics of each prediction. It also computes some statistics on the group's predictions.



**Alternatively, inference.ipynb** could be used to segment the provided sample data with the included checkpoints.

# CNN Architecture

We use the standard U-Net for this work. The only change is the numper of input channels that is variable depending on the set of input channels.

## CNN description

Segmentor(
  (model): UNet(
    (encoder1): Sequential(
      (enc1conv1): Conv2d(5, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc1norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc1relu1): ReLU(inplace=True)
      (enc1conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc1norm2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc1relu2): ReLU(inplace=True)
    )
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (encoder2): Sequential(
      (enc2conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc2norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc2relu1): ReLU(inplace=True)
      (enc2conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc2norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc2relu2): ReLU(inplace=True)
    )
    (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (encoder3): Sequential(
      (enc3conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc3norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc3relu1): ReLU(inplace=True)
      (enc3conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc3norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc3relu2): ReLU(inplace=True)
    )
    (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (encoder4): Sequential(
      (enc4conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc4norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc4relu1): ReLU(inplace=True)
      (enc4conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (enc4norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (enc4relu2): ReLU(inplace=True)
    )
    (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (bottleneck): Sequential(
      (bottleneckconv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bottlenecknorm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bottleneckrelu1): ReLU(inplace=True)
      (bottleneckconv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bottlenecknorm2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bottleneckrelu2): ReLU(inplace=True)
    )
    (upconv4): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
    (decoder4): Sequential(
      (dec4conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec4norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec4relu1): ReLU(inplace=True)
      (dec4conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec4norm2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec4relu2): ReLU(inplace=True)
    )
    (upconv3): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
    (decoder3): Sequential(
      (dec3conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec3norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec3relu1): ReLU(inplace=True)
      (dec3conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec3norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec3relu2): ReLU(inplace=True)
    )
    (upconv2): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    (decoder2): Sequential(
      (dec2conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec2norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec2relu1): ReLU(inplace=True)
      (dec2conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec2norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec2relu2): ReLU(inplace=True)
    )
    (upconv1): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
    (decoder1): Sequential(
      (dec1conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec1norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec1relu1): ReLU(inplace=True)
      (dec1conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (dec1norm2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dec1relu2): ReLU(inplace=True)
    )
    (conv): Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1))
  )
)


## CNN summary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 144, 144]           1,440
       BatchNorm2d-2         [-1, 32, 144, 144]              64
              ReLU-3         [-1, 32, 144, 144]               0
            Conv2d-4         [-1, 32, 144, 144]           9,216
       BatchNorm2d-5         [-1, 32, 144, 144]              64
              ReLU-6         [-1, 32, 144, 144]               0
         MaxPool2d-7           [-1, 32, 72, 72]               0
            Conv2d-8           [-1, 64, 72, 72]          18,432
       BatchNorm2d-9           [-1, 64, 72, 72]             128
             ReLU-10           [-1, 64, 72, 72]               0
           Conv2d-11           [-1, 64, 72, 72]          36,864
      BatchNorm2d-12           [-1, 64, 72, 72]             128
             ReLU-13           [-1, 64, 72, 72]               0
        MaxPool2d-14           [-1, 64, 36, 36]               0
           Conv2d-15          [-1, 128, 36, 36]          73,728
      BatchNorm2d-16          [-1, 128, 36, 36]             256
             ReLU-17          [-1, 128, 36, 36]               0
           Conv2d-18          [-1, 128, 36, 36]         147,456
      BatchNorm2d-19          [-1, 128, 36, 36]             256
             ReLU-20          [-1, 128, 36, 36]               0
        MaxPool2d-21          [-1, 128, 18, 18]               0
           Conv2d-22          [-1, 256, 18, 18]         294,912
      BatchNorm2d-23          [-1, 256, 18, 18]             512
             ReLU-24          [-1, 256, 18, 18]               0
           Conv2d-25          [-1, 256, 18, 18]         589,824
      BatchNorm2d-26          [-1, 256, 18, 18]             512
             ReLU-27          [-1, 256, 18, 18]               0
        MaxPool2d-28            [-1, 256, 9, 9]               0
           Conv2d-29            [-1, 512, 9, 9]       1,179,648
      BatchNorm2d-30            [-1, 512, 9, 9]           1,024
             ReLU-31            [-1, 512, 9, 9]               0
           Conv2d-32            [-1, 512, 9, 9]       2,359,296
      BatchNorm2d-33            [-1, 512, 9, 9]           1,024
             ReLU-34            [-1, 512, 9, 9]               0
  ConvTranspose2d-35          [-1, 256, 18, 18]         524,544
           Conv2d-36          [-1, 256, 18, 18]       1,179,648
      BatchNorm2d-37          [-1, 256, 18, 18]             512
             ReLU-38          [-1, 256, 18, 18]               0
           Conv2d-39          [-1, 256, 18, 18]         589,824
      BatchNorm2d-40          [-1, 256, 18, 18]             512
             ReLU-41          [-1, 256, 18, 18]               0
  ConvTranspose2d-42          [-1, 128, 36, 36]         131,200
           Conv2d-43          [-1, 128, 36, 36]         294,912
      BatchNorm2d-44          [-1, 128, 36, 36]             256
             ReLU-45          [-1, 128, 36, 36]               0
           Conv2d-46          [-1, 128, 36, 36]         147,456
      BatchNorm2d-47          [-1, 128, 36, 36]             256
             ReLU-48          [-1, 128, 36, 36]               0
  ConvTranspose2d-49           [-1, 64, 72, 72]          32,832
           Conv2d-50           [-1, 64, 72, 72]          73,728
      BatchNorm2d-51           [-1, 64, 72, 72]             128
             ReLU-52           [-1, 64, 72, 72]               0
           Conv2d-53           [-1, 64, 72, 72]          36,864
      BatchNorm2d-54           [-1, 64, 72, 72]             128
             ReLU-55           [-1, 64, 72, 72]               0
  ConvTranspose2d-56         [-1, 32, 144, 144]           8,224
           Conv2d-57         [-1, 32, 144, 144]          18,432
      BatchNorm2d-58         [-1, 32, 144, 144]              64
             ReLU-59         [-1, 32, 144, 144]               0
           Conv2d-60         [-1, 32, 144, 144]           9,216
      BatchNorm2d-61         [-1, 32, 144, 144]              64
             ReLU-62         [-1, 32, 144, 144]               0
           Conv2d-63          [-1, 2, 144, 144]              66
             UNet-64          [-1, 2, 144, 144]               0
================================================================
Total params: 7,763,650
Trainable params: 7,763,650
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.40
Forward/backward pass size (MB): 128.30
Params size (MB): 29.62
Estimated Total Size (MB): 158.31
----------------------------------------------------------------

