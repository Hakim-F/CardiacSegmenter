# CardiacSegmenter

This repository contains pretrained U-Net models for the segmentation of anatomical structures of the heart on CINE short-axis, LGE, T1, Post T1, T2, and AO flow images. The code used to train these models is an edited version of the [acdc_segmenter](https://github.com/baumgach/acdc_segmenter).

The related paper is available in open access [here](https://rdcu.be/cjA0f)

If you have any question regarding this repository, send them to this [email](mailto:mhakim.fadil@gmail.com)


## Region of interest
* CINE : Left Ventricle endocardium, Left Ventricle myocardium, Right Ventricle endocardium
* LGE : Left Ventricle endocardium, Left Ventricle myocardium, Scar tissue
* T1, Post T1, T2 : Left ventricle endocardium, Left ventricle myocardium
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

Where `inputImage.nii.gz` is a 4D nifti cine short-axis image.

The segmentation output:`inputImageSeg.nii.gz`

### LGE 

run 

``` python CardiacModels/Unet2DLGE.py -image inputImage.nii.gz ```

Where `inputImage.nii.gz` is a 3D nifti LGE image.

The segmentation output: `inputImageMyoScarSegSelectedRF.nii.gz`


### T1 - Post T1 - T2

run 

``` python CardiacModels/Unet2D_T1_PostT1_T2.py -sequence Seq -image inputImage.nii.gz ```

Where `Seq` is the sequence of `inputImage.nii.gz` : either `T1`, `PostT1` or `T2`

Where `inputImage.nii.gz` is a 3D nifti T1, PostT1 or T2 image.

The segmentation output: `inputImageSegSelected.nii.gz`

example: ```python CardiacModels/Unet2D_T1_PostT1_T2.py -sequence PostT1 -image image.nii.gz ```

### AO FLOW

run 

``` python CardiacModels/Unet2DFLOW.py -image inputImage.nii.gz ```

Where `inputImage.nii.gz` is a 4D nifti AO flow magnitude image (1 slice with multiple frames).

The segmentation output: `inputImageSeg.nii.gz`


## Running training

The code for training is within the `Training` directory of this repository.

Open the `config/system.py` and edit all the paths there to match your system.

The `data_root` directory should contain the images with the ground truth segmentations with the following naming convention:

image: `name_image.nii.gz`  groundtruth: `name_image_gt.nii.gz`

By following this convention, the script will be able to recognize the image and the groundtruth. 

The image and the groundtruth should be a 3D nifti image.

In case of a 4D CINE short axis image, you should convert it to a 3D image, by stacking the frames.
Have a look at `dataManagement.py` for more information.

If you are training with a new dataset dont forget to delete the `preproc_data` directory created for the previous dataset.

Next, open `train.py` and, at the top of the file, select the experiment for the modality you are considering.

To train a model simpy run:

``` python train.py ```

WARNING: When you run the code on CPU, you need around 12 GB of RAM. Make sure your system is up to the task. If not you can try reducing the batch size, or simplifying the network. 

In `system.py`, a log directory was defined. By default it is called `training_log`. You can start a tensorboard
session in order to monitor the training of the network(s) by typing the following in a shell with your virtualenv
activated

``` tensorboard --logdir=training_log --port 8008 ```

Then, navigate to [localhost:8008](localhost:8008) in your browser to open tensorboard.

## References

 In case you find the training code useful, don't hesitate to give appropriate credit to it by citing the related paper, 

 ```
@article{baumgartner2017exploration,
  title={An Exploration of {2D} and {3D} Deep Learning Techniques for Cardiac {MR} Image Segmentation},
  author={Baumgartner, Christian F and Koch, Lisa M and Pollefeys, Marc and Konukoglu, Ender},
  journal={arXiv preprint arXiv:1709.04496},
  year={2017}
}
```
Paper describing this multi-scan pipeline:

Fadil, H., Totman, J.J., Hausenloy, D.J. et al. A deep learning pipeline for automatic analysis of multi-scan cardiovascular magnetic resonance. J Cardiovasc Magn Reson 23, 47 (2021). https://doi.org/10.1186/s12968-020-00695-z


Paper describing the U-Net architecture:

 ```
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

