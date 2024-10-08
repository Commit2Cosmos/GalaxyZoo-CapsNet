# GalaxyZoo-CapsNet

## Table of contents
* [Project Overview](#general-info)
* [Required Packages and Technologies](#technologies)
* [Data](#setup)
* [Files](#files)
* [Acknowledgments](#acknowledgments)

## Project Overview
This project investigated the abilities of a CapsNet to classify galaxy images based on [Galaxy Zoo 2](https://data.galaxyzoo.org) vote fractions and [Simard et al. catalogue](https://ui.adsabs.harvard.edu/abs/2011ApJS..196...11S/abstract) structural parameters. This repository provides code that can process the required image data into the necessary data frame, train both models, evaluate their classification accuracies and test their abilities to reproduce well known physical results related to galaxy evolution.
	
## Required Packages and Technologies
The required packages for this project are:
* Pytorch
* CUDA
* Torchvision
* Sys
* SKimage
* SKlearn
* PIL
* Scipy
* Matplotlib
* Seaborn
* Numpy
* Pandas

This project used 4 CPUs and 1 Tesla V100 GPU on the Lancaster University [High End Computing (HEC) Cluster](https://answers.lancaster.ac.uk/display/ISS/High+End+Computing+%28HEC%29+help).


## Files
### CapsNet
The CapsNet folder contains four versions of the capsule network: ```CapsNetPredictor_2.py```, ```CapsNetPredictor_all.py.py```, ```CapsNetRegressor_2.py``` and ```CapsNetRegressor_all.py```. Each splits the dataset into a training (80%) and testing (20%) sample.

```CapsNetRegressor_2.py``` is used to train the capsule network to predict the Galaxy Zoo vote fractions or structural parameters corresponding to an image. It accepts data in the form of a tensor [Number of images, Number of colour channels, Image width, Image height] and matches each image in the tensor, by index, to the image label tensor [Number of images, Number of parameters]. The network uses an Adam optimizer to minimize the mean squared error between the actual parameters and the network predicted parameters. It outputs the accuracy across all images at each epoch, as well as saves the trained set of weights to the epoch_%d.pt file.

The ```CapsNetPredictor.py``` allows you to load in the pre-trained weights from the epoch_%d.pt file to predict the vote fractions corresponding to a set of input images to the network. 

```CapsNetReconstructor.py``` trains a capsule network to classify a galaxy image as either smooth and rounded, featured or an artefact using binary labels rather than vote fractions. Using the trained set of weights, the network reconstructs the images giving a visualisation of the features that the capsule network is able to detect to classify images. 


Note:

* If training or classifying RGB images change ‘in_channels = 3’, if using greyscale images set ‘in_channels =1’.

* ```CapsNetPredictor.py``` code failed to work when classifying 1 image with it outputting a 16-dimensional vector rather than the predicted array of vote fractions. However, it works fine when classifying more than one image.


### DataAnalysis
**AccuracyPlot.py**

Converts the mean squared error at each epoch into an accuracy and plots how the classification accuracy of a model varies against the number of epochs it is ran for.

**ColourBar_Plot.py**

Plots the Galaxy Zoo vote fraction for a given class against the CapsNet (or ResNet) predicted vote fraction. Each plotted point is colour coded such that the colour represents its Sersic index.

**Colour_Mass_Plot.py**

Creates a colour-mass plot for a sample of galaxies. Instead of representing the galaxy distribution as a scatter plot, contour lines which denote the density of point on the scatter plot are used.

**HistogramPlot.py**

Plots the distribution of galaxies in a sample by their vote fraction for a given class. Particularly useful when trying to match the distribution of DECaLS and SDSS image datasets for consistency.

**KS_Test.py**

Performs both the KS test and Anderson-Darling test between two datasets to determine whether the two datasets are drawn from the same parent distribution. In this particular example, galaxies that are red in colour and classified as smooth by the Galaxy Zoo are compared to galaxies that are red in colour and classified as smooth by the CapsNet.

**ROC_BinaryLabel.py**

Creates an ROC curve using Galaxy Zoo classifications that have been rounded into a binary form, while the CapsNet predictions are left in float form. In the example provided all Galaxy Zoo classifications are rounded to a 0.5 threshold, meaning vote fractions above 0.5 are taken as 1 and those below are taken as 0. The ROC curve is the created as normal with the CapsNet vote fractions being rounded at each possible threshold and then compared to these fixed Galaxy Zoo labels.

**ROC_Plotter.py**

Creates an ROC curve that doesn't require either the Galaxy Zoo or CapsNet predictions to be rounded prior. Instead, both sets of vote fractions are simultaneously rounded and compared at every single rounding threshold.

**ReconstructImages.py**

Plots the reconstructed images from ```CapsNetReconstructor.py``` into a grid format to make comparisons between different epochs easier.

**SersicVotes_Errors.py**

Plots the difference between the Galaxy Zoo and CapsNet vote fraction for a given classification. This difference is then plotted against Sersic index to demonstrate how the classification error of the CapsNet varies with Sersic index. Each point is assigned a colour (red, blue or green) depending on its location on the colour-mass plot. 

### Dataloader
The code within the Dataloader folder is used to convert a folder of images into a suitable tensor that can be fed as an input of image data to either the CapsNet or ResNet. 

For both the ```Segmenter_Dataloader.py``` and ```Dataloader.py``` a directory that will contain all the galaxy images must be specified (‘root_dir=’), as well as the relative file paths/names of each image in that directory. The CSV file, loaded as ‘csv_file=’, must have the first column containing the relative file paths/names of all the images in the image directory. It is also useful to have the other columns in this CSV file corresponding to the vote fractions for each galaxy image. A series of transforms are applied to each image in order to crop them and convert them into a tensor format. Each image tensor is then appended to a list, such that each entry in the list corresponds to a single image tensor. The end result is an .npy file containing all the image data with a shape: [Number of images, Number of colour channels, Image width, Image Height].

The ```Segmenter_Dataloader.py``` works in an identical manner, expect the transforms applied to each image also includes Otsu’s thresholding method which acts to remove the image background.



### Miscellaneous


### ResNet
The ResNet folder contains 4 files: ```ResNetRGB.py```, ```ResNetRGBPredict.py```, ```ResNetGrey.py``` and ```ResNetGreyPredict.py```.

```ResNetRGB.py``` and ```ResNetGrey.py``` are the two sets of code used to train the ResNet model in a similar manner to the ```CapsNetRegressor.py```. Both ResNet models include data augmentation such as a series of horizontal and vertical flips, as well as 45-degree rotations, which effectively quadruples the size of the dataset. The trained weights are saved into the ‘epoch_.pt’ file at the end, these pre-trained weights can then be loaded into either ```ResNetRGBPredict.py``` or ```ResNetGreyPredict.py``` to obtain the predicted vote fractions corresponding to a galaxy image (whether that image is an RGB or greyscale image).
 

## Acknowledgments
Project Acknowledgments

The capsule network used in this project was based on and adapted from the model create by [Reza Katebi et al.](https://arxiv.org/abs/1809.08377) which is available [here](https://github.com/RezaKatebi/Galaxy-Morphology-CapsNet).