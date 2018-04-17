# Lecture Four | April 9 2018

'Super convergance' allows rates of 1-3, which is much higher than normal.

http://gh-demo.kubeflow.org/

Neural Translation (subset of machine translation)
Statistical machine translation is more common / older.
Neural machine translation appeared first in 2015, and wasn't very good. Now, it's significantly better.

Attention in LSTM: To be learned today

Three things needed for a neural net:
1. Data (often (x,y) pairs)
1. Architecture
1. Loss function

For object detection, all of the interesting parts were in the loss function

For neural translation, all of the interesting parts are in the architecture

This is highly based on Lesson 6 (RNNs)

Sequence to sequence with pretrained language models is a promising new area that is yet untouched

`pip install git+[link to github repo here]` very useful!

ex: `pip install git+https://github.com/facebookresearch/fastText`

A ModelData object just holds train/val sets in a single convenient object

Number of rows in an embedding = Size of vocab

In order to use pretrained encoder (backbone), you must use the same length embeddings as they did (which makes sense).

`fastai.core.togpu = False` sometimes for debugging purposes?

Simple sequence to sequence models with low amounts of data can be surprisingly effective

## Post Break

### Trick 1: Bidirectional

Go bidirectional! Take all tokens, spin them around, train a new model.

Add `bidirectional=True` to your GRU encoder (and investigate under the hood)

This slows it down though, as you might imagine. In the Google translate network, only the first layer is bidirectional

### Trick 2: Teacher Forcing

In order to jump start the training process, feed the network the correct answers occasionally

Implementing teacher forcing in TF/Keras made Jeremy switch to PyTorch

If random > pr_force, set the decoder input to be the actual right answer. Set pr_force to be pretty big initially, and then slowly lower it. The language model start out as garbage, and this helps speed up this process.

### Trick 3: Attention (big trick)

Expecting the entirety of the sentence of be summarized into a single hidden vector is asking a lot.

Read: Attention and Augment Recurrent Neural Networks by colah

I'll have to return to this concept, I don't fully understand it -- he's using an additional neural net to help the process out.

## Bringing together text and images

`Devise.ipynb`

We have word encodings. What if we had image encodings as well?

https://en.wikipedia.org/wiki/Word2vec

`import nmslib` is an excellent (and rather unknown) nearest neighbors library

The Devise paper allows you to do all sorts of very impressive things -- text->image search, image->image search
