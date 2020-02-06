Build a Traffic Sign Recognition Program
Udacity - Self-Driving Car NanoDegree Project

Overview
---
In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will then try out your model on images of German traffic signs that we find on the web.

The code is in an Ipython notebook names `Traffic_Sign_Classifier.ipynb`.

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


### Dataset and Repository

The data set can be dowloaded from here ([download the dataset](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)). This is a pickled dataset from MNIST in which the images are resized to 32x32. It contains a training, validation and test set.

The pickled data is a dictionary with 4 key/value pairs:

- `features` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `labels` is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
- `sizes` is a list containing tuples, (width, height) representing the original width and height the image.
- `coords` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. These coordinates assume the original image. The pickled data contains resized versions (32 by 32 of these images).



### Dataset Exploration
##### Dataset Summary
The number of training, validation, and testing samples and image data shape and number of classes in the data set is as follows:   

```Ipython
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

##### Exploratory Visualization
Below we can see one sample image for each of the 43 classes with the name of the class written on top of the image.

![Signs][image-data-exploration-signs]

Let's look at the number of data for each label in training, validation and test sets. As we can see there are not equal number of data for each label.   
![By Label][image-data-exploration-num-by-label]


### Design and Test a Model Architecture
##### Preprocessing
We preprocess the data by doing the following steps:

- **Shuffle**: We shuffle the data so the ordering of the data doesn't affect how well the network trains.
``` Python
X_train, y_train = shuffle(X_train, y_train)
```

- **Convert to grayscale**: Since grayscale images only have one channel we convert each image to grayscale to reduce the amount of data that needs to be processed and trained. We may loose some color information, but this can help the network to train faster.
``` Python
gray_image = cv2.cvtColor(X_train[i,:,:,:], cv2.COLOR_RGB2GRAY)
```

- **Normalize**: We normalize data so that the data has mean zero and equal variance. We used `(pixel - 128) / 255` as a quick way to approximately normalize the data. Normalization makes convergence much faster for the training.
``` Python
XX_train[i,:,:,0] = gray_image / 255.0 - 0.5
```

##### Model Architecture
My model is based off the original LeNet convolution neural networks.

| #   | Layer          | Input Size | Output Size | Description    |
| ----| -------------  | -----------| ------------|:-------------:|
| 1   | Convolution    | 32x32x1    | 28x28x6     | kernel_size=(5,5), RELU|
| 2   | Max Pooling    | 28x28x6    | 14x14x6     | strides=(2,2)|
| 3   | Convolution    | 14x14x6    | 10x10x16    | kernel_size=(5,5), RELU|
| 4   | Max Pooling    | 10x10x1    | 5x5x16      | strides=(2,2)      |
| 5   | Flatten        | 5x5x16     | 400         |    |
| 6   | Fully Connected| 400        | 120         | RELU   |
| 7   | Dropout | 120        | 120         | dropout=0.3   |
| 8   | Fully Connected| 120        | 84         | RELU   |
| 9   | Dropout | 84        | 84         | dropout=0.3   |
| 10  | Fully Connected| 84         | 43         | RELU   |


Here is how the model is defined in the cell # 16 in the Ipython notebook:
```Python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPool2D(strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=84, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=43, activation = 'softmax')
])
```

##### Model Training

Here is a description of how I trained the mode:
- **Optimizer**: To train the model, I chose Adam optimizer. This method is more sophisticated than the other methods such as SGD and it is computationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of data/parameters.
- **Batch Size**: I set the batch size to 128. Batch size tells TensorFlow how many training images to run through the network at a time. The larger the batch size the faster our model will train, but our processor may have a memory limit on how large a batch it can run.
- **Number of Epochs**: I set the number of epochs to 25 to allow the network to pass the target validation accuracy of 93%. In general the more epochs, the better our model trains, but longer the training takes.

Final value for parameters:
```Python
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.001
```

The model was trained with the following line in cell # 18 in the IPython notebook.
```Python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


The final accuracy with the above parameters is as follows:

| #   | Dataset      | Accuracy |
| ----| -------------| -----------|
| 1   | Training     | 98.3%    |
| 2   | Validation   | 94.7%    |
| 3   | test         | 93.4%    |


The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

##### Solution Approach

My approach to solve the traffic sign classification was to use LeNet architecture that has proven to be a highly accurate solution to these type of image classification problems and apply adjustments to improve the accuracy of the model.  

There are various aspects to consider when to increase the accuracy of our model:
- Neural network architecture: The validation accuracy was about 5% lower than the training accuracy. This suggests that the model if overfitting to the training data. To address this, I added to dropout layers after the fully connected layers. This improved the validation accuracy.
- Improve preprocessing techniques: My preprocessing includes normalization, converting images from rgb to grayscale, and shuffling.
- Number of examples per label: Since some classes have way more samples than others, this may result in model bias towards classes with higher samples. We can put a cap on the max number of samples we use from each class for training the model. I didn't apply this in the current version of the code.   
- Generate fake data: To make the model more general we can increase the size of the training set by generating multiple images from each image by applying zooming, shifting, scaling, or changing brightness. I didn't apply this in the current version of the code.   

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

[EDIT]
Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

[EDIT]
their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without.

### Strategies to Improve Classification Model
- experiment with different network architectures, or just change the dimensions of the LeNet layers
- add regularization features like drop out or L2 regularization to make sure the network doesn't overfit the training data
- tune the hyperparameters
- improve the data pre-processing with steps like normalization and setting a zero mean
- augment the training data by rotating or shifting images or by changing colors
