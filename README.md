# caffeModel2hdf5
## Install and convert caffemodel to  hdf5 
1. copy the folder to your caffe directory and run the install script
2. use your network caffemodel and output path to modify the test script 
3. run the test script to generate the hdf5 file

## Example
When you name the layers of Torch model, you should name them as that from caffe model \*.prototxt file.
And you must be careful that the names of caffe model shouldn't contains some special symbol of paths such as "/". Try 
to replace them with "\_".

Plz refer to t googlenet.lua to load weights and bias.

## source 
The project comes from Kencoken. And dump_network_hdf5.cpp is copy from it. The Link is (https://github.com/kencoken/caffe-model-conver)

