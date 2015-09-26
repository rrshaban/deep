# pycaffe

$ ls
caffe/        # python caffe wrapper. This folder lets us `import caffe` if we want to
models/       # stores model snapshots (.caffemodel, .solverstate)
nets/         # Network definitions  (.prototxt)

Datasets are stored on `/scratch/rshaban1/lmdb/`.

### Things you might run

`python generate_prototxt.py > output.prototxt`

`caffe.bin train -solver net_solver.prototxt`

`caffe.bin test -solver solver.prototxt -weights iter_4000.caffemodel.h5 -model train_test.prototxt`