# Lecture One | March 19 2018

Check out playground.tensorflow.org

Arch design is not that interesting -- there's a few that tend to work well and we generally stick to that.

Arch design is generally not the hard part.

Overfitting and how to avoid it:
- More data
- Data aug
- Generalizable architectures (dropout / weight decay)
- Regularization
- Reduce arch complexity (less layers, less activations) (why?)

Step one to not overfitting is overfitting (ensure the data set / arch actually works)
If you don't start at 'overfitting', you're lost.

Embeddings: NLP or any categorical data can be modeled using NNs and embeddings.
'Tabular data'

## Practical -> Cutting Edge

Part 1 was best practices, and reliable techniques. We learned stuff Jeremy had used a lot and had put into the fastai lib.

Part 2 is cutting edge deep learning for coders. Not practical deep learning anymore.

He no longer knows all the answers. Lessons will be incomplete, with ideas to explore.

However, he only will be teaching stuff that's promising and worth learning about.

No longer will fastai be a black box.

He'll introduce python debugger, and how to use Vim to jump around :)

More detailed code walkthroughs

More details paper walkthroughs

## What do I do after each lecture?

"How to use the provided notebooks":

Don't open up the provided notebook and hit shift enter until a bug appears.

The notebook is a CRUTCH. The idea is you start with an empty notebook, and think "I want to complete X process". This might involve you reading the code from the notebook, but DO NOT copy and paste it. Type it out. Make sure you can type it out. Understand it.

Hopefully you can figure out *what* everything is doing. However, also try to figure out *why* we're doing things.

Get an ASUS GTX 1070 for $350?

A graphics card 3x faster than a K80 is $600.

Get a 1080ti if you can afford it. 1070 is also fine.
Get 32GB RAM is possible.

CPU speed matters when doing CV. The speed of the data aug is often the bottleneck, and data aug happens on the CPU. Or, if hard drive can't give data to GPU fast enough.

x8 PCI lanes is enough per GPU.

## Opportunities

Our homework will be cutting edge -- talk about it

Experiment a lot

Start before you're ready

Stay in your domain -- it makes it easier to experiment as you're already familiar with the problems, you're already interested, you already have the data.

Write about your work despite imposter syndrome

Write something that would have helped you 6 months ago!

## Part 2 Overview

Part 2 is about Generative Models

In part I, the output of our NNs was *a* number, or *a* category.

The outputs of NNs in part II will be a lot of things.
Ex:
- The input paragraph, translated to French
- X and Y coords of an object + its category
- etc

We're mostly looking at text or image data.

***

After break...

## Object Detection

We have bounding boxes + categories

### Step 1: Classify and localize the largest object* in each image

\*where object is within a predetermined objects that we feed the classifier

Using the pascal dataset.

`torch.cuda.set_device(1)` specifies which GPU to use if your system has multiple GPUs. Zero indexed. This is better(?) that just running the model on all GPUs on the system?

We'll use the 2007 version of the dataset. 2012 version is a bit better. Often people combine them (but there's overlap so be careful)

`Path()` is part of Python3 stdlib. It gives you object oriented access to a directory or a file. It is very useful.

`PATH.iterdir()` returns a generator.

`list(PATH.iterdir())` -> creates a list from the generator

They overrode the division operator, very cool:
`(PATH/'filename.txt').open()`

How to enable tab completion for strings:
`IMAGES = 'images'` (not really but you get the idea, handy for long category names that you have to type a lot)

The difference between an effective individual vs not is *tenacity*. Deep learning has a more difficult feedback reward loop than regular programming. It's a lot more "it doesn't work" than regular programming.

Jeremy's philosophy: The more your eye can see in one view, the more you can understand. Jeremy tries very hard to reduce vertical height but also horizontal height. Some very large, very old programming communities use this approach with success.

In math, everything is a single character. Jeremy finds this less than ideal.

In Java, variable names can be very very long. Jeremy also finds this less than ideal.

`Row x Col` vs `Width x Height` can be confusing. Fastai is always `Rows x Col`.

#### Lots of helper functions for plotting images

We're defining bounding boxes as top left coord and bottom right coord.

I should probably download Anaconda and vscode.

OpenCV is about 5x faster than torchvision.

PIL is not as fast as OpenCV, and not nearly as thread safe. Python has a global interpreter lock (GIL). Python basically can't do multithreading.

Fastai uses multiple threads not multiple processes, which makes it a lot faster. However, OpenCV documentation is horrible.

Apparently matplotlib has an OO API that no one actually uses? Cool.

Add little functions to your notebooks as you use them -- once you use it in 3+ notebooks, add it to a library.

We need to get the largest bounding box in an image -> create a helper function

Dict comprehensions are great

Visually examine each stage of your pipeline

A data loader is an interator. Everytime you grab the next batch, you get a minibatch. By default, batch size is 64.

Running each line of code, printing all inputs and outputs, is a good way to understand what's going on inside a block of code.

`pdb` is excellent.

If an exception occurs:
`%debug` opens the deubgger at the point the exception occurred

In CSV, labels must be space separated
