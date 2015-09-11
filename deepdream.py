# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

# test if cuda is set up on oil
caffe.set_mode_gpu()
caffe.set_device(0)

# def showarray(a, fmt='jpeg'):
#     a = np.uint8(np.clip(a, 0, 255))
#     f = StringIO()
#     PIL.Image.fromarray(a).save(f, fmt)
#     display(Image(data=f.getvalue()))
