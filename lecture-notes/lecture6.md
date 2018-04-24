# Lecture Six | April 23 2018

Fine Tuned Language Models for Text Classification, Jeremy's paper, got accepted.

Image Enhancement is the topic of today's lesson

Differential learning rates is now being called "Discriminative learning rates"

https://github.com/luanfujun/deep-painterly-harmonization
https://medium.com/@hortonhearsafoo/adding-a-cutting-edge-deep-learning-training-technique-to-the-fast-ai-library-2cd1dba90a49
https://arxiv.org/abs/1801.06146

Gradually increasing the data size (resolution of images, etc) is useful for fast training

CNN activations are a rank 4 tensor(?)

https://colah.github.io/posts/2014-10-Visualizing-MNIST/
https://github.com/sgugger/Deep-Learning/blob/master/Understanding%20the%20new%20fastai%20API%20for%20scheduling%20training.ipynb

"Factored convolutions" -- very interesting concept to speed up conv nets

Continually increase image size throughout training a GAN
https://arxiv.org/abs/1710.10196
https://github.com/tkarras/progressive_growing_of_gans
http://research.nvidia.com/publication/2017-10_Progressive-Growing-of very impressive demo!

Dropout is patented by Geoffrey Hinton. 

There's an ImageNet sample on files.fast.ai/data

---

Instead of using the gradient to update the model weights, we're now updating the pixels

Input random noise, use the loss function to slowly make it more like pics of birds / paintings of Van Gogh
Two other inputs: pics of birds, and Van Gogh paintings


Two different loss functions:
* Content Loss: returns a value that's lower if it looks more like a bird 
* Style loss: returns a value that's lower if it looks more like Van Gogh's work

We could just do MSE using the pytorch optimizer to turn the random noise image into a bird image. This is a good exercise. 

However, there's an issue here -- how does style loss work? How do we figure out if an image looks "like" Van Gogh? Answer: use a neural net. Specifically, today we will use VGG. So, we use a NN to abstract away the specifics of the image (beak, wings, etc) and then use that.

Content loss will be what's called "perceptual loss", which just means comparing two activations together. 

"Gaddys style transfer" is what this is called

Loss function for style transfer:
His code is here: https://github.com/VinceMarron/style_transfer
His paper is here: https://github.com/VinceMarron/style_transfer/blob/master/style-transfer-theory.pdf

Jeremy found it was very difficult to use a loss fxn to turn a picture of random noise into an image. Then, he tried a more blurred img (median_filter), and it worked better. Pictures are smooth, random noise isn't. 

Intro to BFGS: (initials of 4 different people)
Uses gradients (as do most optimizers that we use)
It does a bit more work than say, SGD
(Hessian = Gradient derivative)
BFGS calculates the Hessian, and uses that to figure out what direction to go, and how far to go. Less of a wild jump into the unknown.
Sadly, calculating the Hessian isn't a great idea, as it's very computationally expensive. 
Instead, we simply take a few steps and observe the change in the gradient, and approximate the Hessian
So:
* Keep the last 10-20 steps
* Use that to calculate the Hessian
* Take action based on that 

If you want a half precision Volta network to train properly, you generally need to multiply the loss function by a scaler (512 or 1024 etc)

