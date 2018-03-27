# Lecture Two | March 26 2018

<!-- <><><><><><><> -->

We are continuing object detection (category + bounding box)

Research fast ai custom head inside a Learner

Regression vs Classification: Continuous vs Discrete

`learn.summary()` is useful to see how data looks as it's passed through the model

Three things needed to train a neural net:
- Data (usually lots)
- An architecture (to pass the data through)
- A loss function (to perform backprop with)

### Data:

We need a data set with the input (independant variable) being an image, and the output (dependant variable) being a tuple of the category + coords of bounding box

There's a bunch of different ways to have a dataset with two different dep variables -- Jeremy shows one way in his notebook.

### Architecture

We can use the same arch as the classifier and bounding box regression, we just combine them.

Example: if we have `C` classes, we need `4 + C` outputs (activations) from the neural net. (4 = # of data points to define a bounding box). Then, the `C` activations can be used to determine which category the object is in.

### Loss Function

The loss function needs to look at the `4 + C` activations and decide "are these good?". For the first 4, we use L1 loss (which is like MSE, but it's sum of abs value, not sum of squares).

Activations = 'input'
Ground truth = 'target'

*Architecture Tip: Put RELU and then batch norm in your layers*

(Sometime Jeremy breaks this rule to be consistent with a paper or something -- it's not a huge deal)

Figuring out what the main object in an image is is kind of the hard part. Figuring out where it is and what class it is is a bit easier. If you have a single network that figures out where the object is AND what the object is, there's a lot of shared computation.

## Pascal Multi

Apparently pandas can be used in place of a default dict?

There's two approaches for multi object classification:

1. Output a massive long vector (used by YOLO)
1. Output a rank 3 tensor (`4x4x(4+C)` for 16 different anchor boxes), which leverages inherent features of a conv network (used by SSD)

The second approach is becoming recognized as superior -- YOLO3 uses it. This is because of something called a 'Receptive Field' in ConvNets (easiest shown in a drawing -- there's a lot of good stuff on Google Images).

A lot of discussion about anchor boxes vs bounding boxes -- there's a lot of history around Object Detection. Five key papers:

1. Scalable Object Detection using Deep NNs
1. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
1. YOLO: Unified, Real Time Object Detection
1. SSD: Single Shot Multibox Detector
1. Focal Loss for Dense Object Detection

The 2nd paper uses a two pass method, first to get a "suggested region" and then another to get the classifications.

Paper 3 & 4 figured out how to do the same thing, but in only one stage.

Focal Loss is fairly new, late 2017. They realized why small objects are so hard to detect. There are three different granularities of boxes: 4x4, 2x2, 1x1. 1x1 pretty much always has an overlap with an interesting subject. However, the majority of 4x4 boxes have no subject in them. It's a good bet that most of the boxes are just 'background'.

There's this parameter, gamma. In the paper, they discuss how well classified examples have 0.6+ (ish) probability of ground truth class. But, the loss is still 0.5 or so. In order to fix this, you can tune gamma such that the loss is more like zero when the probability is 0.6.

Essentially, you can tell the network that 60% confidence is "good enough" in order to classify an object. Because small objects are so often not objects, the network learns to just never classify them as objects. Focal Loss is how you fix this (gamma tuning specifically).

```
Focal Loss = Cross Entropy Loss * (1-p)^gamma
```


Multiplying the loss by `(1-p)^gamma` was a huge realization. It's very subtle, but it fixes a huge problem that everyone was trying to solve.

It's important to know what cross entropy is and how it's defined.

CE(p,y) = -log(p) if y=1 else -log(1-p)

Then, they refactor (see the paper).

This fixes this greatly. However, we still have problems.

Last 10 mins of lecture (rewatch!): great discussion of how to read journal papers.

In the future: Feature Pyramids
