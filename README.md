## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files:
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[image-data-exploration-signs]: ./output_images/data_exploration_signs.jpg "Signs"
[image-data-exploration-num-by-label]: ./output_images/number_data_by_label.jpg "Data by Label"


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

The data set can be dowloaded from here (INSERT LINK). This is a pickled dataset from MNIST in which the images are resized to 32x32. It contains a training, validation and test set.

The pickled data is a dictionary with 4 key/value pairs:

- `features` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `labels` is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
- `sizes` is a list containing tuples, (width, height) representing the original width and height the image.
- `coords` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES



### Dataset Exploration
The number of training, validation, and testing samples and image data shape and number of classes in the data set is as follows:   

```Ipython
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```


##### Dataset Summary

##### Exploratory Visualization
These are 8 random images from the data set with their corresponding label.

![Signs][image-data-exploration-signs]

Let's look at the number of data for each label in training, validation and test sets. As we can see there are not equal number of data for each label.   
![By Label][image-data-exploration-num-by-label]
### Design and Test a Model Architecture

##### Preprocessing
We preprocess the data by doing the following steps:

- **Shuffle**: We shuffle the data so the ordering of the data doesn't affect how well the network trains.
- **Convert to grayscale**: Since grayscale images only have 1 channel We convert each image to grayscale to reduce the amount of data that needs to be processed and trained.
- **Normalize**: We normalize data so that the data has mean zero and equal variance. We used `(pixel - 128) / 128` as a quick way to approximately normalize the data.




(INSTRUCTION: preprocessing techniques used and why these techniques were chosen)


##### Model Architecture
(INSTRUCTION: The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.)

##### Model Training

The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

##### Solution Approach

The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

### Test a Model on New Images
##### Acquiring New Images

The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

##### Performance on New Images

The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.


##### Model Certainty - Softmax Probabilities

The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

##### AUGMENT THE TRAINING DATA
(BONUS) Augmenting the training set might help improve model performance. Common data augmentation techniques include rotation, translation, zoom, flips, and/or color perturbation. These techniques can be used individually or combined.

##### ANALYZE NEW IMAGE PERFORMANCE IN MORE DETAIL
(BONUS) Calculating the accuracy on these five German traffic sign images found on the web might not give a comprehensive overview of how well the model is performing. Consider ways to do a more detailed analysis of model performance by looking at predictions in more detail. For example, calculate the precision and recall for each traffic sign type from the test set and then compare performance on these five new images..

If one of the new images is a stop sign but was predicted to be a bumpy road sign, then we might expect a low recall for stop signs. In other words, the model has trouble predicting on stop signs. If one of the new images is a 100 km/h sign but was predicted to be a stop sign, we might expect precision to be low for stop signs. In other words, if the model says something is a stop sign, we're not very sure that it really is a stop sign.

Looking at performance of individual sign types can help guide how to better augment the data set or how to fine tune the model.

##### CREATE VISUALIZATIONS OF THE SOFTMAX PROBABILITIES
(BONUS) For each of the five new images, create a graphic visualization of the soft-max probabilities. Bar charts might work well.

##### VISUALIZE LAYERS OF THE NEURAL NETWORK
(BONUS) See Step 4 of the Iptyon notebook for details about how to do this.

### Strategies to Improve Classification Model
- experiment with different network architectures, or just change the dimensions of the LeNet layers
- add regularization features like drop out or L2 regularization to make sure the network doesn't overfit the training data
- tune the hyperparameters
- improve the data pre-processing with steps like normalization and setting a zero mean
- augment the training data by rotating or shifting images or by changing colors
