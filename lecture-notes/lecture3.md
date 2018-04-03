# Lecture Three | April 2 2018

Research 'matching problem' in object detection

Inside `pdb.set_trace()`, you can do `var1,var2,var3,...` to print each variable on one line.

Replace NMS in object detection with end to end neural net?

I need to read (and implement) more papers.

When reading a paper, map the math to the code as a way to build your understanding in both domains

When reading a paper, try to recreate the figures that they created.

## NLP time

We've moved from torchtext to fastai.text

fastai.nlp is deprecated and obsolete now.

We're using IMDb again (same as lesson 4).

`from fastai.text import *`

To randomize multiple lists in the same way:

```
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
# add other lists here
```

There's a bit of a standard format for NLP when you put the data in a CSV file

LM = Language Model in an NLP context

Once we have a big long string, we need to tokenize it. We want to tokenize by word... mostly. We want "don't" to be two different words, and we want "." (full stop) to be its own word, etc.

Jeremy has used this exact code on datasets of over a billion words. One trick for bigger dataset: with Pandas, use the chunksize feature to leverage generators instead of returning an entire dataframe.

Lowercasing everything is often a bad idea because you lose data.

However, leaving the cases as is means the neural net needs to learn both "Hello" and "hello", which is a lot of work. Similarly, learning that "!!!!" and "!!!!!" are similar is a lot of work. Jeremy presents several tricks to help the network out.

Convert to indicies
1. Make a list of all words that appear
1. Replace every word with its index in that list

You can't learn anything about a word if it doesn't appear sufficiently often, so enforcing a `min_freq` is valuable. Additionally, having a vocab of >60k words can be an issue, so enforcing a `max_vocab` is also good.

---

IMDb movie reviews are not that different from any other English doc (compared to random string or Chinese). Just like imagenet allowed us to classify things not from imagenet (satellite images, medical images, etc).

Hence: let's train a model that's generally good at English, and then finetune to IMDb. Introducing: the wikipedia dataset.

Jeremy has also trained a model and made it publicly available -- make sure that you use the same embedding size.

Load the model in with `torch.load()`

Problem: the previously mentioned word->index mapping is different for different models.

Solution: interate through the wikipedia embedding matrix and copy it over as needed.

## Create Language Model

1. Concat all docs together into a single list of token (of length 24.9m)
1. Set up some dropout
1. Create a model data object
1. Call the usual `fit()`

*wikitext103:*
400 deep embedding
3 hidden layers
1150 activations per layer

What is `batchify()`?

In NLP, if our batch size is 64, and we have 25m words, we DO NOT create items of legnth 64. We create 64 different items, of length 25m/64.

You can't shuffle the order of words the way you can shuffle images, so here's a neat little trick to change your bptt from Steven Merity et al:

```
bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2
seq_len = max(5, int(np.random.normal(bptt, 5)))
get_batch(seq_len)
```

You can do NLP transfer learning with a backbone + head of a model.

Read: Regularizing and Optimizing LSTM Language Models

Cross entropy loss vs Accuracy

Complexity in NLP is e^(cross entropy loss)

Accuracy is just how often you guess the next work correctly. It's a more stable metric to keep track of.

The rate at which a model can understand language is increasing very rapidly (especially the last 12-18 months), very similar to 2011-2012 CV models.

Multi threading > Multi processing

BPTT can change from training session to training session.

Transfer learning -> Cut state of the art loss by 20%

http://yann.lecun.com/

Read "A disciplined approach to neural network hyper params: Part 1 - learning rate, batch size, momentum, weight decay"

Look into the Google Fire library for iterating through a large amount of Python function parameters from the command line.

Symlink fastai to your `site-packages/` directory to be able to import the library anywhere (similar to putting a binary in `/bin`)

