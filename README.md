# caffeModel2hdf5

##Target 
Load weights and bias from caffe to Torch

1. save caffe model to a common file appended with caffemodel such as bvlc\_googlenet.caffemodel
2. convert caffemodel file to hdf5 file
3. load weights and bias from hdf5 file in Torch

Load input layer data from caffe to Torch
1. after net completes forward operation, the data will be fed to net. And copy data to the dataLayer's top blob
2. save dataLayer's top blob to hdf5 file
3. load weights and bias from hdf5 file in Torch

## Install and convert caffemodel to  hdf5 
1. copy the folder to your caffe directory and run the install script
2. use your network caffemodel and output path to modify the test script 
3. run the test script to generate the hdf5 file

## Example
When you name the layers of Torch model, you should name them as that from caffe model \*.prototxt file.
And you must be careful that the names of caffe model shouldn't contains some special symbol of paths such as "/". Try 
to replace them with "\_".

Plz refer to googlenet.lua to load weights and bias.

## Source 
The project comes from Kencoken. And dump_network_hdf5.cpp is copyed from it. The Link is (https://github.com/kencoken/caffe-model-conver)

