// This program takes in a trained network and dump all the network
// parameters to a HDF5 file that is readable for Mocha.jl
// Usage:
//    dump_network_hdf5 network_def network_snapshot hdf5_output_filename
//
// Please refer to Mocha's document for details on how to use this tool.

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/data_layer.hpp"
using namespace caffe;  // NOLINT(build/namespaces)

void dump_weight_bias(hid_t &h5file, Layer<float> *layer, const string& layer_name, const string& weight_name);
void dump_single_blob(hid_t &h5file, Blob<float>* blob,  const string& blob_name);
int main(int argc, char** argv) {
  caffe::GlobalInit(&argc, &argv);
  Caffe::set_mode(Caffe::CPU);

  if (argc != 4) {
    LOG(ERROR) << "Usage:";
    LOG(ERROR) << "  " << argv[0] << " net-def.prototxt net-snapshot.caffemodel output.hdf5";
    exit(1);
  }

  const char *network_params   = argv[1];
  const char *network_snapshot = argv[2];
  const char *hdf5_output_fn   = argv[3];

  hid_t file_id = H5Fcreate(hdf5_output_fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

 
  shared_ptr<Net<float> > caffe_net;
  caffe_net.reset(new Net<float>(network_params, caffe::TRAIN));
  caffe_net->CopyTrainedLayersFrom(network_snapshot);
  
  const vector<shared_ptr<Layer<float> > >& layers = caffe_net->layers();
  const vector<string> & layer_names = caffe_net->layer_names();
  
  float loss = 0;
  caffe_net->Forward(&loss); //after data layer completes forward operation, it will load a batch and copy data blob to top.

  for (int i = 0; i < layer_names.size(); ++i) {
   
    if (DataLayer<float> *layer = dynamic_cast<DataLayer<float> *>(layers[i].get())) {
      LOG(ERROR) << "Dumping DataLayer " << layer_names[i];
      
      const  vector<vector<Blob<float>* > >& top_vecs_ = caffe_net->top_vecs();
      float blob_sum = (top_vecs_[i][0])->asum_data();
        
      dump_single_blob(file_id, (top_vecs_[i][0]), string("InputLayer"));
    } else if (InnerProductLayer<float> *layer = dynamic_cast<InnerProductLayer<float> *>(layers[i].get())) {
      LOG(ERROR) << "Dumping InnerProductLayer " << layer_names[i];
      dump_weight_bias(file_id, layer, layer_names[i], string("weight"));
    } else if (ConvolutionLayer<float> *layer = dynamic_cast<ConvolutionLayer<float> *>(layers[i].get())) {
      LOG(ERROR) << "Dumping ConvolutionLayer " << layer_names[i];
      dump_weight_bias(file_id, layer, layer_names[i], string("weight"));
    } else {
      LOG(ERROR) << "Ignoring layer " << layer_names[i];
    }
  }

  H5Fclose(file_id);
  return 0;
}

void dump_single_blob(hid_t &h5file, Blob<float>* blob,  const string& blob_name)
{
    if(NULL == blob)
    {
        LOG(ERROR) << "Input layer Error!!!";
        exit(1);
    }
    
    hdf5_save_nd_dataset(h5file, blob_name, *blob, false);

}

void dump_weight_bias(hid_t &h5file, Layer<float> *layer, const string& layer_name, const string& weight_name) {
  vector<shared_ptr<Blob<float> > >& blobs = layer->blobs();
  if (blobs.size() == 0) {
    LOG(ERROR) << "Layer " << layer_name << " has no blobs!!!";
    exit(1);
  }

  LOG(ERROR) << "    Exporting weight blob as '" << weight_name << "'";
  
  hdf5_save_nd_dataset(h5file, layer_name + string("___") + weight_name, *blobs[0], false);
  if (blobs.size() > 1) {
    LOG(ERROR) << "    Exporting bias blob as 'bias'";
    hdf5_save_nd_dataset(h5file, layer_name + string("___bias"), *blobs[1]);
  } else if (blobs.size() > 2) {
    LOG(ERROR) << "Layer " << layer_name << " has more than 2 blobs, are you serious? I cannot handle this.";
  } else {
    LOG(ERROR) << "    No bias blob, ignoring";
  }
}
