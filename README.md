# Diagnosing Pink Eye using Convolutional Neural Networks
Using Convolutional Neural Networks to diagnose conjunctivitis, or pink eye

Please note, this repo is still a work-in-progress, and is in no way fully operational.

**By: Lachi Balabanski**

## Table of Contents:

- [Background](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#background)
  - [Conjunctivits/Pink Eye](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#conjunctivitispink-eye)
  - [Convolutional Neural Network(CNN)](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#convolutional-neural-networkcnn)
  - [Real-life application](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#real-life-application)
- [Use](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#use)
  - [Dependencies](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#dependencies)
  - [Training/Diagnosis](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#trainingdiagnosis)
  - [How Does it Diagnose Me?](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#how-does-it-diagnose-me)
- [CNNs in Depth](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#cnns-in-depth)
  - [The Perceptron](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#the-perceptron)
  - [Backpropagation](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#backpropagation)
  - [Gradient Descent](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#gradient-descent)
  - [Neural Nets](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#neural-nets)
  - [Convolutional Neural Nets](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#convolutional-neural-nets)
  - [Neural Networks' Shortcomings](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#neural-networks-shortcomings)
    - [Overfitting](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#overfitting)
    - [Obvious Equivilents](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#obvious-equivilents)
    - [Processing Power](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#processing-power)
- [Citations](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs#citations)

## Background:

### Conjunctivitis/Pink Eye:
Conjunctivitis, commonly called pink eye, is an inflammitory diesease of the eye. Pink eye does not affect vision, but can be very irritating.
Unfotunately, pink eye is a very common problem. This is why I set out to diagnose it using convolutional neural nets.

### Convolutional Neural Network(CNN):
Convolutional Neural Networks are neural networks that pass an image through a 'filter' or convolution. The neural network determines how
much the image fits the convolution. The CNN has many of these convolutions, too many to program in manually, so instead, the power to modify
its own convolutions is given to the CNN. This is critical to the process, because this means that the program has to undergo training to get
its convolutional weights accurate. To learn more, go to the *CNNS In Depth* section of the readme.

## Real-life application:
The applications of this program and of CNN's are *limitless*. Convolutional Neural Nets are already used in face recognition software, self-driving cars, and object recognition. Most illnesses have some form of visible symptom, i.e. inflamation of eyes, swollen lymph nodes, etc. So computers will finally be able to diagnose paitents, for minor illnesses, without having to go to a doctor. In the future, the technology may be able to look at MRIs, X-rays, CAT scans, etc. and take all of these factors into account and be able to diagnose paitents, simmilar to the way doctors today do.

It is important to note that, even though the use of CNNs may be fun to experiment with, a computer still cannot replace a real doctor's diagnosis. **Do not take advice given to you by the program as a real medical diagnosis**

## Use:

### Dependencies:
This program uses numpy and tensorflow, which are not built-in. Be sure to pip install both to use this repo

### Training/Diagnosis:
To start, change the current working directory to the saved directory of the repo. Then, copy and paste all images to be dianosed to the *'to_diagnose'* folder Execute this in the terminal:

```bash
$ python cnn_main.py -n [names of images]
```

*-n* or *--names* expects a comma-seperated list of image names with the extension(*i.e. test1.jpg,test2.jpg,test3.jpg*)

If it was succesful, the training should display something along the lines of:

```text
Loading Data. . .   [DONE]
100 images accepted
0 images were not suitable
================
Epoch: 1, Loss: 0.9908674, Accuracy: 0.119856
3% finshed
================
Epoch: 20, Loss: 0.778223, Accuracy: 0.213785
61% finished
===============
```

Eventually the program should output:

```text
Optimization is done!
Image 'test1.jpg' has a 2% chance of having pink-eye
Image 'test2.jpg' has a 97% chance of having pink-eye
Image 'test3.jpg' has a 1% chance of having pink-eye
```

The program will shortly display its diagnosis of pink eye on the images in the to_diagnose folder, specified by the -n or --names parameter.

### How Does it Diagnose Me?
Through training, the computer is able to recognize patterns in pink eye. Compare these two images, for example:

**Image 1, regular eyes** | **Image 2, pink eye**
:-------------------------|----------------------:
<img src='https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs/blob/master/img/regular_eye.jpg' width=200 height=200> | <img src='https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs/blob/master/img/pink_eye.jpg' width=200 height=200>

It is easy for us to tell which one has pink eye, and which one does not. The computer, at first, has a little bit of trouble. We must train it to asscosiate certain features of the face(i.e. swollen eyes, red-colored eye, etc.) with pink eye. Simmilar to how we learn, the computer gets better as it sees more examples of pink eye.

## CNNS In Depth:

### The Perceptron:
The perceptron is the basic unit of a neural net. Simmilar to a cell in biology or an atom in chemistry, it is the smallest functioning unit of a neural network. Perceptrons work by inputing values called weights and biases and outputing a summation of the dot products of the weights and biases. The computer takes this sum, and puts it through an activation function. Activation functions introduce non-linearities to our machine's best fit of the data. Non-linearities are important, because most real-world data is very complex to model with linear functions. The three current most popular activation functions are:

1. ReLu: ReLu is a piece-wise activation function which outputs one when the input is greater than zero, and it outputs one when the input is less than or equal to zero

2. Sigmoid: Sigmoid is an activation function which outputs a number betweeon zero and one

3. Hyperbolic Tangent: Hyperbolic tangent can return complex values

Each has their own advantages and disadvantages.

During training, the biases are adjusted by a process called backpropogation.

### Backpropagation:
Invented by Geoffery Hinton, backpropagation, or backprop for short, is a method of finding the difference between the returned result and the expected result and subtly adjusting biases of nodes that contributed to the gap in expected and returned results.

### Gradient Descent:
To minimize empirical loss, we want to use a gradient descent optimizer. Empirical loss can be thought of as an approximation for how close the predicted result is to the actual result. We want to minimize this quantity. So, we take a step of some magnitude in a random direction, and we measure the descent or ascent. If we're going up, we want to go the other way. Otherwise, keep going down. The procedure that was just described is a very simple gradient descent algorithm.

There are many gradient descent algorithms, the most popular of which include: Momentum, Adagrad, and Adam.

The size of the magnitude of the step we took is called the learning rate. The learning rate is important, because if it is set too high or too low, it can get the program stuck in a local minimum for the value of the empirical loss, rather than a global optimum.

### Neural Nets:
Neural Nets are called the way they are, because they mimic the human brain, *neural*, and net is short for network. The network in a neural net is a series of perceptrons linked together.

![Alt text](https://github.com/lachibal2/diagnosing-pink-eye-with-CNNs/blob/master/img/neural_net.jpg) 

Neural Nets are a just a large collections of perceptrons. Neural Nets have three main types of layers:

1. input layers: the input data,

2. output layers: what the computer thinks is the answer, and

3. hidden layers: where the bulk the computations happen.

Neural Nets with few hidden layers are called shallow neural networks and ones with many hidden layers, upwards of hundreds, are called deep neural networks. The process of using deep neural networks to create intelligent, informed decisions is called deep learning.

### Convolutional Neural Nets:
As previously stated, CNNs pass images through a filter, called a convolution. Convolutions take in a small section of the image and look at the small section for how much each pixel of the small section matches each convolution pixel. The convolution returns a value for how much the small section resembles the convolution. This value can then be put through another convolution, after another, etc.

The problem is, there are too many layers for a programmer to go through and program each one. As previously stated, there could be upwards of hundereds of layers, each many nodes tall. Instead, Neural Nets are used. Neural Nets allow the program to add to or modify its current structure, by changing the biases.

### Neural Networks' Shortcomings:

#### Overfitting:
Overfitting to the test data is when a computer "memorizes" the data set, rather than making generalizations for it. The program will recognize test images very well, but will work poorly on new images, or generating new images.

To counter overfitting, two solutions have been developed. The first is dropout. In dropout, a portion of the cells in each layer are not used for one iteration. This allows the program to not focus on any one cell. The other technique is to look at when loss starts to increase. The point loss on testing images increases is a good point to stop training your neural net.

#### Obvious Equivilents:
Neural nets work very well for images that are in a simmilar orientation as the ones they trained on. Unfortunately, a neural net is not yet capable of saying, "This is the number '4', but rotated 90 degrees."

#### Processing Power:
This is less of a shortcoming of neural networks, than a shortcoming of modern day computers. The computers of today do not have enough processing power to consider enough variables to make the problem interesting.

This is why quantum computing will likely be monumental in AI and deep learning research. By increasing the speed at which we do calculations (from O(n^2) to polynomial time, for some algorithms), we will finally be able to compute more interesting problems.

## Citations:
*Pink eye (conjunctivitis). (2017, June 20). Retrieved January 21, 2018, from https://www.mayoclinic.org/diseases-conditions/pink-eye/symptoms-causes/syc-20376355*

*Unsupervised Feature Learning and Deep Learning Tutorial. Retrieved January 21, 2018, from ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/*

*Convolutional Neural Networks (CNNs / ConvNets). CS231n Convolutional Neural Networks for Visual Recognition. Retrieved January 21, 2018, from cs231n.github.io/convolutional-networks/*
