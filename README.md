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



Alternatively, inference.ipynb could be used to segment the provided sample data with the included checkpoints.

