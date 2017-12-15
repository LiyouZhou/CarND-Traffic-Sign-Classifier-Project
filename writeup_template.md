# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyse the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### Basic summary of the data set

- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

#### Exploratory visualization of the dataset.

<img src="examples/histogram_of_classes.png" width="500">
The above image shows the distribution of data among the different classes of signs. You can see there are very few data sets for certain classes of traffic signs compared to others which will no doubt impact the accuracy of the learning of those classes later.

<img src="examples/random_sample_from_each_class.png" width="500">
Here is a matrix of random images pulled out of each class. The first thing that can be noted is that the images appear to be taken under a large variety of lighting conditions. This does add to the complexity of the classification.

### Design and Test a Model Architecture

#### Data pre-processing

##### Generate extra data

Because there are significantly more samples of some classes than others in the training set, the model accuracy will have a bias toward the classes with more samples. Bad accuracy in a classes with a lot of samples leads to a large loss. In order to make sure the model recognise all classes of traffic sign equally well, more data is generated. This is achieved by taking a available image and randomly zoom, rotate and translate to result in an new image. An example can be seen below:

<img src="examples/augmented_image.png" width="400">

##### Grayscaling

The image is turned into greyscale because all information needed to recognise a traffic sign is encoded into the shape of the sign, colour varies a lot with different lighting condition. Getting rid of the colour component reduces the complexity of the module and reduces the irregularities in the data.

Here is the traffic sign images from the previous section after grayscaling.

<img src="examples/greyscale_normalised.png" width="400">

##### Histogram Equalisation

This is an technique to even out the lighting condition of all the data further reducing data irregularities.

Here is the traffic sign images from the previous section after grayscaling.

<img src="examples/equalised.png" width="400">

It can be seen that images that are previous completely dark and hard to recognise and been brightened up and the whole dataset have the appearance of uniform lighting.

##### Normalisation

Make input data normally distributed with mean 0 and std 1. This allow weights and hyper-parameters to stay in predictable range and makes training and tuning to be faster.

#### Final model architecture

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 Greyscale image                       |
| Convolution           | 1x1 stride, same padding, outputs 32x32x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution           |                                               |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fully connected       | Input = 400. Output = 120                     |
| RELU                  |                                               |
| Dropout               | 50% dropout probability                       |
| Fully connected       | Input = 120. Output = 84                      |
| RELU                  |                                               |
| Dropout               | 50% dropout probability                       |
| Fully connected       | Input = 84. Output = 10                       |

This is a LeNet architecture with two dropout layers for regularisation.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper-parameters such as learning rate.

starting from a BATCH_SIZE of 128 and a learning rate of 0.001, it is found that increasing number of epochs improved validation accuracy while changing learning rate in either direction made very little difference. BATCH_SIZE of 32 and a learning rate of 0.001.

The number of EPOCHS to run is constrained by the time and computing resources available. 20 EPOCHS was run and the final model was taken at the epoch where the validation accuracy was highest. This is an early termination regularisation where the model is taken at the highest accuracy point during the training process. In theory, if more time was available one could run more EPOCHS and could probably obtain a higher accuracy model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

A LeNet architecture is chosen because it is a proven model for learning image based data sets. The convolution layers can recognise features on various scales and is independent of the location of the feature inside the image.

It is observed during training that the testing data set accuracy is lower than the validation data accuracy. This suggest possible over-fitting. Hence the dropout layers are added as additional regularisation. If the drop rate of the layer is set too high, the model learns too slowly. A final rate of 0.5 is chosen for balance of training speed and regularisation effectiveness.

My final model results were:
* training set accuracy of 0.979
* validation set accuracy of 0.970
* test set accuracy of 0.941

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over-fitting or under-fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### Five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

<img src="examples/5-images-from-net.png" width="400">

All the images have been padded to square and resized to 32x32 pixels. The 2nd and 5th image might be difficult to recognise because they occupy very small portion of the whole image. The 4th image has a glare which might cause difficulty as well.

Here are the results of the prediction:

|Image                                   |Prediction                              |
|:--------------------------------------:|:--------------------------------------:|
|Dangerous curve to the right            |Children crossing                       |
|No vehicles                             |No entry                                |
|Beware of ice/snow                      |Right-of-way at the next intersection   |
|Beware of ice/snow                      |Road work                               |
|Speed limit (80km/h)                    |Speed limit (20km/h)                    |
|Slippery road                           |Speed limit (30km/h)                    |

The accuracy is zero. We must have a problem. By examining the training data set and the images above it is found that the training images are framed such that the traffic sign is centred and covers most of the image. Hence we need to re-frame the images found online in order for them to recognised. By manually cropping the images, we have the following:

<img src="examples/5-images-from-net-cropped.png" width="400">

The model was then able to currently recognise 3 out of the 5 pictures giving an accuracy of 60%.

|Image                                   |Prediction                              |
|:--------------------------------------:|:--------------------------------------:|
|Right-of-way at the next intersection   |Children crossing                       |
|No entry                                |No entry                                |
|Beware of ice/snow                      |Right-of-way at the next intersection   |
|Speed limit (80km/h)                    |Road work                               |
|Speed limit (30km/h)                    |Speed limit (30km/h)                    |


By calculating the softmax probabilities of the model output we can see the confidence of the predictions. Image 2 and 5 were predicted correctly with near 100% confidence. The top 5 softmax probabilities of the 2 wrong predicitons can be seen below.

|Image 1 (Children crossing                    )|Image 4 (Road work                  )|
|:---------------------------------------------:|:-----------------------------------:|
| 61.41% (Wild animals crossing                )| 62.64% (Speed limit (80km/h)       )|
| 33.56% (Right-of-way at the next intersection)| 12.25% (Road narrows on the right  )|
|  3.02% (Slippery road                        )|  9.80% (End of speed limit (80km/h))|
|  1.76% (Children crossing                    )|  8.45% (Speed limit (30km/h)       )|
|  0.12% (Road work                            )|  6.20% (Wild animals crossing      )|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
