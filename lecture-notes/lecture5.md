# Lecture Five | April 16 2018

## cifar-darknet

Leaky RELU rarely hurts, and often helps a bit. YOLO3 uses it.

It often helps the most on small datasets.

`inplace=False` in the LeakyReLU allocates a whole bunch of memory unnecessarily. Use `inplace=True`
for more efficient computation and higher batchsizes. Also don't forget `add_` and other underscore functions which are inplace.

BatchNorm has bias built in -- so you don't need a bias in a previous convnet. (You need one or the other)

1x1 conv is basically just taking the dot product of an image, multiplying each pixel by all the filters you're using. It doesn't change the width or height of the image (this is fairly intuitive).

SELU self normalizes, so you don't need batchnorm. However, it's very finnicky and hasn't really gone anywhere.

`AdaptiveAvgPool2d()` is better than `AvgPool2d()`. AvgPool2d is tied to a specific architecture size. Adaptive allows you to ask for an output, agnostic of image input size.

Volta has support for fp16, which is half support floating numbers. Fastai is the first library that Jeremy knows of that supports half precision.

Leslie's One Shot -> Investigate.
> Momentum starts high, dips low halfway through, and then jumps back up.

> Learning rate starts high, jumps up halfway through, and then drops back down. At the very tail end, it then drops down to basically 0 (last 15% of epochs in this case).

## WGAN

Generative Adversarial Networks!

Generative networks try to create stuff. It will try to create stuff that is undetectable from real stuff. Face swapping, the answer to a medical question, etc.

The idea behind adversarial networks is that you have a generator and a "discriminator" (also called a critic). The generator tries to create better and better content, and a discriminator that tries to get better and better at detecting fakes.

We'll be using a GAN to create images of bedrooms.

The specific bit about a WGAN is that the parameters are kept inside a certain small range(?)

### Step One: Build a discriminator

A discriminator takes in an image, and spits out a number. The number will be lower if it thinks the image is fake.

Transfer Learning + GANs? So far largely unexplored?

Siraj Raval made a video on GANs: https://www.youtube.com/watch?v=yz6dNf7X7SA

### Step Two: Build a generator

> Mode Collapse: one of the problem with GANs

Mode Collapse is when your GAN 'collapses' down to a set number of 'modes'. In the "create a bedroom image" example, this would be when a GAN only spits out, say, 4 different templates of bedroom. This is easy to spot, but what if it was 10,000 different templates of bedrooms? That's pretty hard to detect as you're training the network. This is still a large problem, and often papers sweep the issue under the rug.

You can use cycle GANs to generate artwork.

Interesting translation paper: https://arxiv.org/abs/1711.00043

Cool conv gifs: http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html

pytorch cgan: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix



