# Google's deepdream python code, with exceptional commentary from 
# http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/

# Other useful resources to explore: 

# http://overstruck.com/how-to-customize-googles-deepdream/


# CURIOUS ABOUT LAYERS?

# Caffe's layers documentation: 
# http://caffe.berkeleyvision.org/tutorial/layers.html

# Beautiful visualization of the effects of each layer:
# https://www.reddit.com/r/deepdream/comments/3clppv/animations_showing_convergence_of_deepdream/

# Step-by-step illustration of the layers:
# http://hideepdreams.com/post/123387228638/testing-layers-of-googles-deepdreams

# Curious why the layers are called Inception? 
# http://arxiv.org/pdf/1409.4842.pdf


from cStringIO import StringIO          #     Used to save and display the image in the IPython notebook 
                                        #      as its generated, used only in save_image()

import numpy as np                      #     Used to do all the matrix math, with the exception of the zoom

import scipy.ndimage as nd              #     Used for nd.zoom on the images (octaves) in deepdream()
import PIL.Image                        #     Used to load images from file, and to save manipulated images back to 
                                        #     files

from IPython.display import clear_output, Image, display
                                        #     Used to display the images as the neural runs over iterations

from google.protobuf import text_format #     Inter language data format for our blobs. A way to save and load trained
                                        #     networks in Caffe. Only used to add our 

import caffe                            #     The machine learning framework upon which everything works.

from datetime import datetime           #     for timestamping

caffe.set_mode_gpu()                    #     Uncomment to put computation on GPU. You'll need caffe built with 
caffe.set_device(0);                    #     CuDNN and CUDA, and an NVIDIA card


model_root = '/scratch/rshaban1/models/'

m = {
    'bvlc_googlenet':       ('bvlc_googlenet/', 'bvlc_googlenet.caffemodel', 'deploy.prototxt'),
    'googlenet_places205':  ('googlenet_places205/', 'googlelet_places205_train_iter_2400000.caffemodel', 'deploy.prototxt'),
    'vgg_16':               ('vgg_16/', 'VGG_ILSVRC_16_layers.caffemodel', 'VGG_ILSVRC_16_layers_deploy.prototxt'),
    'vgg_19':               ('vgg_19/', 'VGG_ILSVRC_19_layers.caffemodel', 'VGG_ILSVRC_19_layers_deploy.prototxt'),
    }

odel = 'vgg_16'
model_path = model_root + m[odel][0]
param_fn   = model_path + m[odel][1]
net_fn     = model_path + m[odel][2]      # specifies the neural network's layers and their arrangement
                                                    # we load this and patch it to add the force backward below

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".

model = caffe.io.caffe_pb2.NetParameter()           # Load the empty protobuf model,
text_format.Merge(open(net_fn).read(), model)       # Load the prototxt and load it into empty model
model.force_backward = True                         # Add the force_backward: true line
open('tmp.prototxt', 'w').write(str(model))         # Save it to a new file called tmp.prototxt

net = caffe.Classifier('tmp.prototxt', param_fn,    # Load the neural network. Using the prototxt from above
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0))      # the reference model has channels in BGR order instead of RGB

# The above code loads up our neural network, but there are a few interesting things to note. 

# -> param_fn - our trained neural network blob

# -> mean - the RGB average of our images. we will later subtract and then add this back to our model.
#           interesting changes can be made by changing these numbers, and for the places dataset, the
#           numbers above are actually wrong. I couldn't find the proper mean for it. [I couldn't either - RS]

# -> channel_swap - different order of values, with Blue, Green and Red as the matrix order. switches it
#                   to Red, Green, Blue
 
# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    # rolling the axis switches the structure of the matrix
    # We go from having Red, Green, and Blue values for each x and y coordinate
    # to having a 3 channels of red, green and blue images.

# Now this function above is a doozie. Don't let its shortness deceive you, it's doing a lot.
# net.transformer.mean['data'] is our image mean we discussed above. it's being subtracted from
# np.float32(np.rollaxis(img, 2)[::-1]). This function warrants a discussion about numpy. I'll 
# dive deeper into it below. [He discusses this with examples in the article - RS]

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# This function does the reverse of preprocess, and I'll go over how it works below too. But 
# both these functions warrant talking about how they work.

def preprocesswithoutmean(img):
    return np.float32(np.rollaxis(img, 2)[::-1]) # see preprocess() for explanation

def deprocesswithoutmean(img):
    return np.dstack(img[::-1])


def objective_L2(dst):          # Our training objective. Google has since released a way to load
    dst.diff[:] = dst.data      # arbitrary objectives from other images. [Explanation of this in the article - RS]


def save_image(a, f='out', fmt='jpeg', out_path='out/'):
    
    a = np.uint8(np.clip(a, 0, 255))    #     Convert and clip our matrix into the jpeg constraints (0-255 values
                                        #     for Red, Green, Blue)
    if f == 'out':
        f = datetime.utcnow().isoformat().replace(':','_')

    out = out_path + f.replace('/','_') + '.jpg'    # allows us to feed either the layer name or timestamp as filename

    PIL.Image.fromarray(a).save(out)

    # display(Image(data=f.getvalue())) #     Display the image in our notebook, using the IPython.display, and 
                                        #     IPython.Image helpers.


def make_step(net, step_size=1.5, end='inception_4b/3x3',      
              jitter=32, clip=True, objective=objective_L2):    #  This end layer differs from the original Google deepdream
    '''Basic gradient ascent step.'''                           #  in order to match the places dataset

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]    # destination is the end layer specified above

                            # The end layer is the name of the layer or blob (as some call it) at which to stop
                            # For explanation of the layer structure of the model, check out
                            # http://overstruck.com/how-to-customize-googles-deepdream/

                            # Step-by-step illustration of the layers:
                            # http://hideepdreams.com/post/123387228638/testing-layers-of-googles-deepdreams

    # Jitter is well-explained in the article with illustrative examples. Jitter just
    # shifts the image over a few pixels with some degree of randomness
    # jpg/exploring.jpg + jitter = jpg/fixed_exploring.jpg

    ox, oy = np.random.randint(-jitter, jitter+1, 2)            # generate random jitter
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    # Forward and backward passes: http://caffe.berkeleyvision.org/tutorial/forward_backward.html

    net.forward(end=end)     # This is how the network computes, make sure we stop on the chosen neural layer
    objective(dst)           # specify the optimization objective
    net.backward(start=end)  # Do backwards propagation, so we can compute how off we were 
    g = src.diff[0]          
    
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g # get closer to our target data

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image jitter
            
    if clip:                                              # If clipping is enabled
        bias = net.transformer.mean['data']               # Subtract our image mean
        src.data[:] = np.clip(src.data, -bias, 255-bias)  # clip our matrix to the values



def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4b/3x3', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]   # So, the octaves is an array of images, initialized 
                                            # with the original image transformed into caffe's format
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    # Okay, so this creates smaller versions of the images, and appends them to the array of images.
    # One image for each octave.
    
    src = net.blobs['data']             # Again, copy the original image.
    detail = np.zeros_like(octaves[-1]) # Allocate image for network-produced details.
                                        # This creates a matrix shaped like our image, 
                                        # but fills it with zeroes.

    for octave, octave_base in enumerate(octaves[::-1]): # Iterate over the reversed list of images (smallest first)
        h, w = octave_base.shape[-2:]                    # Take the width and height of the current image
        if octave > 0:  # If it's not the smallest octave
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1) # Zoom in on the image detail, interpolate

        src.reshape(1,3,h,w)                                    # resize the network's input image size
        src.data[0] = octave_base+detail                        # Add the changed details to the image
        for i in xrange(iter_n):                                # number of step iterations, specified above
            make_step(net, end=end, clip=clip, **step_params)   # call the function that actually runs the network
            
            print octave, i, end #, vis.shape

            # # visualization - this will sometimes be commented out if we don't want output every octave
            # vis = deprocess(net, src.data[0])       # Convert back to jpg format
            # if not clip:                            # adjust image contrast if clipping is disabled
            #     vis = vis*(255.0/np.percentile(vis, 99.98))
            
            # save_image(vis)
            
        # extract details produced on the current octave
        detail = src.data[0] - octave_base

    # returning the resulting image
    return deprocess(net, src.data[0])

frame = np.float32(PIL.Image.open('jpg/ap.jpg'))

print net.blobs.keys()

for i in net.blobs.keys()[1:]:      # every layer in the system
    if "split" not in i:            # except layers like '{}/output_0_split_0', which break for some reason
        save_image(deepdream(net, frame, octave_n=4, end=i), f=i, out_path='out/vgg_16/')

# save_image(deepdream(net, frame, octave_n=2, end='inception_5a/3x3_reduce'), f='inception_5a/3x3_reduce')

# _ = deepdream(net, frame, end='inception_3b/5x5_reduce')
# _ = deepdream(net, img, end='inception_3b/5x5_reduce')





