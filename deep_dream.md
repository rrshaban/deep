Deep Dreams


Documentation for Caffe in Python is [sparse](https://github.com/BVLC/caffe/issues/1774) and a bit of a [work in progress](https://github.com/BVLC/caffe/pull/1703). To try to get a feel for how to use Caffe, I went looking for examples of pyCaffe in the wild and found Google's [Deep Dream](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html), which they put on [Github](https://github.com/google/deepdream) with some spotty commenting. KP Kaiser put together [a pretty phenomenal walkthrough of that code](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/), which is the basis for [my annotated version](https://github.swarthmore.edu/DeepLearningCS93/pycaffe/blob/master/deepdream.py) of the deep dream code.

This led to exploring and trying to understand the Deep Dream code in order to reverse-engineer it. Here's the BVLC GoogleNet Deep Dream @ [2 octaves](http://imgur.com/a/i4CBW) and at [6 octaves](http://imgur.com/a/w3xsz). 

Google's Deep Dream uses the [Inception architecture](http://arxiv.org/pdf/1409.4842.pdf), a deep architecture. Here's what the GoogleNet deepdream architecture looks like in [Caffe's YAML format](https://github.swarthmore.edu/DeepLearningCS93/pycaffe/blob/master/models/bvlc_googlenet/deploy.prototxt)

[What an MNIST autoencoder in Caffe looks like](https://github.com/BVLC/caffe/blob/master/examples/mnist/mnist_autoencoder.prototxt)

