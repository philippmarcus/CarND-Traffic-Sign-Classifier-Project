# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/output_11_0.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/output_40_0.png "Traffic Signs from the Web"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image10]: ./examples/output_14_0.png "Traffic Signs Overview"
[image11]: ./examples/output_16_0.png "Image created from mean values"
[image12]: ./examples/output_24_0.png "Overview of preprocessed data"
[image13]: ./examples/output_53_0.png "Prediction analysis for web data"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799

* The size of the validation set is 4410

* The size of test set is 12630

* The shape of a traffic sign image is (32, 32)

* The number of unique classes/labels in the data set is 43


The data set covers the following traffic signs:
![alt text][image10]

#### 2. Include an exploratory visualization of the dataset.


Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

From the bar chart we see, that the number of samples is not balanced across the different classes. This may lead to a biased classificator. Remedy could be to collect more data for the underrepresented classes.

Comparing Test, Valid and Train data set are not quite equal, too. Some classes are under- or overrepresented compared to the other data sets.

### Design and Test a Model Architecture

#### 1. Preprocessing the Image Data

As a first step, I decided to convert the images to grayscale to reduce the size of the training data. As the color is widely considered as not too important, this conversion was assumed to be acceptable.

As a last step, I normalized the image data because to give the pixels in the dataset a zero mean and a standard deviation of 1. The normalization of the data is important for faster convergence of the optimizer and to avoid unprecise values due to floating point characteristics.

This was done, by computing for each pixel the mean value and the standard deviation and normalizing each image by this. For better understanding, I created an example image by the mean values for each pixel in the training dataset:

![alt text][image11]

This image is subtracted from every image in the training data set. Afterwards, it is divided by the standard deviation image.

Here is an overview of the preprocessed data.

![alt text][image12]


#### 2. The final model architecture

The model was highly inspired by LeNet-5. However, I already from the beginning changed the activation function to ReLU, to avoid the vanishing gradients problem.

My final model consisted of the following layers:

|Layer					|ID				|Details                         |Shape of Output
|:-----------------:|:----------:|:------------------------------:|:----------------
|Input					|In				|preproc img with [None, 32, 32] |with shape [None, 32, 32]
|Conv2d				|C1				|padding=SAME, kernel=5, stride=1, ReLu with 0.5|with shape [None, 32, 32, 6]
|Avg Pool				|S2				|padding=SAME, kernel=2, stride=2|with shape [None, 32, 32, 6]
|Conv2d				|C3				|padding=SAME, kernel=5, stride=1, ReLu with 0.5|with shape [None, 32, 32, 16]
|Avg Pool				|S4				|padding=SAME, kernel=2, stride=2|with shape [None, 16, 16, 16]
|Conv2d				|C5				|padding=SAME, kernel=5, stride=1, ReLu with 0.5|with shape [None, 16, 16, 120]
|Flatten				|C5_flatten	|Details                         |with shape [None, 30720]
|Fully Connected		|F6				|Fully connected, batch normalization, ReLu with 0.5,      |with shape [None, 84]
|Fully Connected		|Out			|Fully connected, batch normalization                         |with shape [None, 43]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I defined a operation in the model to compute the cross-entropy for each sample of the batch. It is of course based on a one-hot encoded vector of the true label and the softmax vector of the logits, that were produced by the Layer "Out". These cross entropies for each element of a batch are then reduced by calculating the average root square sum. This is the final loss function that needs to be minimized.

The training itseld was conducted by taking batch sizes of 1000 and doing 5 epochs. Basically, the batch size could be increased according to the memory size of the computer. The number of epochs needs to be adjuste.

A first mistake I made here was not to shuffle the batches in each epoch, which lead to very poor results.

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

Initially, I initialized the weight matrices with a random normal distribution. This lead to extremly poor training results. After changing the initialization to Xavier Initialization, the accuracy on the test set improved much faster in each epoch of the training.

I added a dropout of 0.5, in order to avoid the model from extrem overfitting on the training data. In this case, the dataset is not that big, and thought that it could lead to overfitting. So I considered this step as a good idea.

After that, still the model trained extremly slow. I adjusted the learning rate from 0.01 to 0.001 and got faster progress but still not satisfying. Obviously, the high traning rate "jumped" over the right "valleys" where the minima of the loss function were lying. I also decided to use the Adam Optimizer, as it adapts the learning rate automatically during training to achieve better results faster.

Still, the accuracy could not reach the required level. I decided to implement batch normalization in the two fully connected layers. Basically, the batch normalization normalizes the output ot the matmul and add function, before it is passed to the ReLU. It prevents too extreme values and allows faster changes of behaviour of the ReLU during back propagation, as all values are somewhat centered around 0. However, it adds two more parameters to the model that need to be fitted, which increases computational complexity.

My final model results were:

* training set accuracy of 1.0

* validation set accuracy of 0.96666664 

* test set accuracy of 0.9528899

Obviously, the model highly overfitted the training data. Further methods of regularization could be tried out to avoid this. I could think of the lasso method, to punish too high weights. Also it might be interesting to try out batch normalization on the convolutional layers. In this case, I think adding more training data would also be a very promising option.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet-5 was chosen. Originally, it was designed for recognizing images of hand-written digits.

* Why did you believe it would be relevant to the traffic sign application?
I have chosen this model, as many traffic signs also contain digits and symbols that look similar to digits. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final accuracy on the test set shows that in a certain percentage of predictions, the prediction of the answer is correct. The accuracy on the training and validation data do not provide this proof, because I adapted the model based on results I got with these data. Thus, information leaked into my model.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image4] 

I made it challenging and took pictures of traffic signs, that are underrepresented in the training data set. All images that are underrepresented in the training data might be wrongly classified more likely. If we look on the histogram provided above and on the "average image", we see that the model will be biased. I assumed that things like the roundabout sign or the blue direction sign or the stop sign could be problematic. It turned out true...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			       |     Prediction	        				| 
|:--------------------:|:---------------------------------------------:| 
| Stop Sign      		| No Vehicles 								| 
| Priority road    		| Priority road								|
| Right-of-way at the next intersection					| Right-of-way at the next intersection							|
| Turn left ahead	   | Speed limit (70km/h)					 		|
| Bumpy road				| Road Work   								|
| Go straight or left	| Keep right							|
| Turn right ahead				| Turn right ahead    							|
| General caution				| General caution  							|
| Speed limit (80km/h)				| Speed limit (80km/h)   							|
| Roundabout mandatory				| Speed limit (100km/h) 							|

My assumption turned out true. Underrepresented images were wrongly classified to classes that have a high representation, e.g. The Turn left ahead was classified as Speed limit (70km/h). A similar case happend for the Roundabout mandatory shield, which was classified as Speed limit (100km/h).

It would make sense to have a look at the confusion matrix of the model, to systematicall analyze where the errors happen.

The model was able to correctly guess 5 of the 10 traffic signs, which gives an accuracy of 50%. Initially it was 60%, but I continued to train the model. This lead to overfitting on the test set and reduced the performance on the data from the web. It would be necessary to detect during training, when overfitting starts and then roll-back to the last model that was not overfitted.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The code for making predictions on my final model is located in the last code cell of the Ipython notebook.

For reasons of simplicity, I added the bar chart with the top 5 softmax probabilities to each image. However, the scale is not normalized, so it can only be used to comare the ratio of the single softmax values for each example seperately. The softmax probabilities are visible in the charts.
![alt text][image13]

It is interesting, that the model shows high softmax values for some wrong classifications. This is caused by the bias on the training dataset.

