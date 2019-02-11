# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

I had some problems with the workspace (it was not saved and I had to restore the data). I hope to have uploaded all necessary files again, but yet something might be missing (in data I only uploaded all the center images because I don't use the others, so I hope this is no problem).

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2018_09_11_08_32_39_320.jpg "Grayscaling" 
[image3]: ./examples/center_2018_09_12_09_03_32_365.jpg "Recovery Image"
[image4]: ./examples/center_2018_09_12_09_03_33_628.jpg "Recovery Image"
[image5]: ./examples/center_2018_09_12_09_03_34_540.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model4.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model4.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model4.h5
```

#### 3. Submission code is usable and readable

The model4.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model4.py lines 61-91) followed by fully connected layers. The starting point I used was the NVIDIA network shown  in lesson 15. I adapted this to the different image size we have here and tried changing several things to get the best result.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 65). Additionally I used the cropping of the input images proposed in lesson 14. I tried adding batch normalization but I got better results without it.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I also tried adding batch normalization but I got better results without it.

The model was trained and validated using a generator (line 26-53) on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model4.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in the other direction, using the other track and flipping the images. I tried using my data and when the car did not stay on the track I tried to analyze the problem and add more data to prevent this.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the NVIDIA example and adapt it to the current situation (different image size) in a way that training does not take too long (not too many trainable parameters), but the result is still good enough.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because because it was used for image recognition before. But as the results were not really good I switched to the NVIDIA architecture presented in lesson 15 trying to improve.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Whenever I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set I tried increasing the dropout or adding batch noremalization to prevent overfitting. 

 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especial the first sharp curve caused me considerable problems. To improve the driving behavior in these cases, I used a combination of adding more data (especially the curves and returning to  the center of the road) and trying to adapt the architecture (batch size, dropout, layer-sizes, number of layers etc.).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24)  consisted of a convolution neural network with the following layers and layer sizes: After cropping and preprocessing I had three convolutional layers with 5x5 kernel, strides of 2, relu activation, Dropout of 0.5 and sizes 24,36 and 48. These were followed by two convolutional layers of sizes 48 and 64 with 3x3 kernel (rest as above). After flattening I  added four fully connected layers with relu activation    and dropout (except the last one) of sizes 100, 50, 10 and 1.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded some laps on track one using center lane driving (as good as I could). Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to behave the way I wanted when it is not in the center. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated the process of center driving on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help learning to treat left and right curves equally well. 


After the collection process, I had X number of data points. I then preprocessed this data by cropping the image and afterwards normalizing to the range between -.5 and .5.


I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1. I used an adam optimizer so that manually training the learning rate wasn't necessary.
