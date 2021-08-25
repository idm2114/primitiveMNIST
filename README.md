# Pytorch basic MNIST handwritten number classifier

## The why? 
So that I can practice Pytorch and learn some basics.

## The how? 
To be determined.

## Last last time

* Worked on getting the basic classifier working
* At the end of the stream, got everything basic working
* Accuracy was hovering around 65% after 5 epochs

## Last time  

* How can we improve accuracy? 
* Variation of parameters for learning rate and more
    * Changed learning rate to 1e-2 and attained 83% accuracy after 5 epochs

## Goal for Today

* Parse an image that contains multiple numbers (using second neural net?)
    * Isolate each number one by one
    * Pass each individual image to our existing neural net 
    * let it make a prediction

## Current progress 

* Created a neural network using a basic CNN (convolutional neural network)
    * The network has 98% accuracy on data in the test dataset
    * In practice, seems to be slightly worse, but oh well
    * We are currently able to take an image of a set of numbers written by 
        a human and parse them one by one with fairly good results

    * Example: 404-937-3661 
    * we recover 304-932-3121, so 60% accuracy in realworld application
