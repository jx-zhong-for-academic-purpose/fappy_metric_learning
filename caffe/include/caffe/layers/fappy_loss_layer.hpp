//
// Created by zjx on 17-1-22.
//

#ifndef CAFFE_FAPPY_LOSS_LAYER_HPP_
#define CAFFE_FAPPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template<typename Dtype>
class FAPPYLossLayer : public LossLayer<Dtype> {
  public:
    explicit FAPPYLossLayer(const LayerParameter &param)
        : LossLayer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top);

    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
        const vector<Blob<Dtype>*> &top);

    virtual inline const char *type() const { return "FAPPYLoss"; }

    virtual inline int ExactNumTopBlobs() const { return 1; }

    virtual inline int ExactNumBottomBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
        const vector<bool> &propagate_down,
        const vector<Blob<Dtype>*> &bottom);
  
  private:
    inline void ElementwiseUpdate(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top,
        const Dtype& k_delta,
        const int& mini_batch_size);
};

}


#endif //CAFFE_FAPPY_LOSS_LAYER_HPP_
