# Overview
## CardiacSegmenter

This repository contains pretrained U-Net [reference] models for the segmentation of anatomical structures of the heart on CINE short-axis, LGE, T1, Post T1, T2, and AO flow images. The code used to train these models is an edited version of the acdc_segmenter [reference].

#Anatomical structure segmented
* CINE : Left Ventricle endocardium, Left Ventricle myocardium, Right Ventricle endocardium
* LGE : Left Ventricle endocardium, Left Ventricle myocardium, Scar tissue
* T1 - Post T1 - T2 : Left ventricle endocardium, Left ventricle myocardium
* FLOW : Aorta

# Installation

## Getting the code

Clone the repository by typing

``` git clone https://github.com/Hakim-F/CardiacSegmenter.git ```

## Requirements 

- Python 3.4 (only tested with 3.4.3)
- Tensorflow >= 1.0 (tested with 1.1.0, and 1.2.0)
- The remainder of the requirements are given in `requirements.txt`

## Installing required Python packages

Create an environment with Python 3.4. If you use virutalenv it 
might be necessary to first upgrade pip (``` pip install --upgrade pip ```).

Next, install the required packages listed in the `requirements.txt` file:

``` pip install -r requirements.txt ```

Then, install tensorflow:

``` pip install tensorflow==1.2 ```
or
``` pip install tensorflow-gpu==1.2 ```

depending if you are setting up your GPU environment or CPU environment. The code was also
tested with tensorflow 1.1 if for some reason you prefer that version.

WARNING: Installing tensorflow before the requirements.txt will lead to weird errors while compiling `scikit-image` in `pip install -r requirements`. Make sure you install tensorflow *after* the requirements. 
If you run into errors anyways try running `pip install --upgrade cython` and then reruning `pip install -r requirements.txt`. 

## Pretrained segmentation models

After cloning the repository on your computer, download the pretrained models and unzip them in the `CardiacModels/Models` directory.

* CINE https://drive.google.com/open?id=1qdAU6XXJzh9HVPXz4kvZHndxCxwbO64z
* FLOW https://drive.google.com/open?id=122f5wGBGcpB8neM8iUpWbZWyYjYX0Rz9
* LGE https://drive.google.com/open?id=1Kn5Yp8SgOMZ3ZRcetMhtKrI4ZjO6LMfb 
* POST T1 https://drive.google.com/open?id=1vzuxZkw7yaNR0CItdtAuZRsbRAULPcYD
* T1 https://drive.google.com/open?id=1VUH_yVSrpIluk0wbcmPE6JQUcOpUUPXn
* T2 https://drive.google.com/open?id=1RvxBQ6VxNO0vpm5HnL87-lVKomKRJfVe

# Usage

## Running segmentation scripts with pretrained models

After downloading the pretrained models and unzipping them in `CardiacModels/Models`

### CINE

run 

``` python CardiacModels/Unet2DCINE.py -image inputImage.nii.gz ```

Where `inputImage.nii.gz` is 4D nifti cine short-axis image.

The segmentation output will be `inputImageSeg.nii.gz`

### LGE 

run 

``` python CardiacModels/Unet2DLGE.py -image inputImage.nii.gz ```

Where `inputImage.nii.gz` is 3D nifti LGE image.

The segmentation output will be `inputImageMyoScarSegSelectedRF.nii.gz`


### T1 - Post T1 - T2

run 

``` python CardiacModels/Unet2D_T1_PostT1_T2.py -sequence Seq -image inputImage.nii.gz ```

where `Seq` is the sequence of `inputImage.nii.gz` : either `T1`, `PostT1` or `T2`
Where `inputImage.nii.gz` is 3D nifti T1, PostT1 or T2 image.

The segmentation output will be `inputImageSegSelected.nii.gz`

example: ```python CardiacModels/Unet2D_T1_PostT1_T2.py -sequence PostT1 -image image.nii.gz ```

### AO FLOW

run 

``` python CardiacModels/Unet2DFLOW.py -image inputImage.nii.gz ```

Where `inputImage.nii.gz` is 4D nifti AO flow magnitude image (1 slice with multiple frames).

The segmentation output will be `inputImageSeg.nii.gz`


## Running training

