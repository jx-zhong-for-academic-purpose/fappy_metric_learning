//
// Created by zjx on 17-1-22.
//

#include "caffe/layers/fappy_loss_layer.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void FAPPYLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num() * (bottom[1]->num() - 1) / 2);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void FAPPYLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}


template <typename Dtype>
inline void FAPPYLossLayer<Dtype>::ElementwiseUpdate(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const Dtype& k_delta,
    const int& mini_batch_size){
  int pos_cnt = 0;
  for(int i = 0; i < mini_batch_size; ++i){
    for(int j = i + 1; j < mini_batch_size; ++j){
      if(bottom[1]->data_at(i, 0, 0, 0) == bottom[1]->data_at(j, 0, 0, 0)){
        const int s_ij_idx = i < j ? mini_batch_size * i - i * (i + 1) / 2 + j - i - 1 : mini_batch_size * j - j * (j + 1) / 2 + i - j - 1;
        const Dtype s_ij = bottom[0]->cpu_data()[s_ij_idx];

        if(1.0 - s_ij < k_delta){
          continue;
        }

        const int hij_left_bound_idx = floor((s_ij + 1.0) / k_delta);
        const Dtype hij_left_bound = k_delta * hij_left_bound_idx - 1.0;
        ++pos_cnt;
        int neg_cnt = 0;
        int h_size = 1 + ceil(2.0 / k_delta);
        std::vector<Dtype> h_ik(h_size, 0);
        std::vector<Dtype> h_jk(h_size, 0);
        std::vector<vector<int> > hik_leftidx_to_cosidx(h_size, std::vector<int>(0));
        std::vector<vector<int> > hjk_leftidx_to_cosidx(h_size, std::vector<int>(0));
        std::vector<vector<int> > hik_rightidx_to_cosidx(h_size, std::vector<int>(0));
        std::vector<vector<int> > hjk_rightidx_to_cosidx(h_size, std::vector<int>(0));
        for(int k = 0; k < mini_batch_size; ++k){
          if(bottom[1]->data_at(i, 0, 0, 0) != bottom[1]->data_at(k, 0, 0, 0)){
            ++neg_cnt;
            const int s_ik_idx = i < k ? mini_batch_size * i - i * (i + 1) / 2 + k - i - 1 : mini_batch_size * k - k * (k + 1) / 2 + i - k - 1;
            const Dtype s_ik = bottom[0]->cpu_data()[s_ik_idx];
            const int hik_left_bound_idx = floor((s_ik + 1.0) / k_delta);
            const Dtype hik_left_bound = k_delta * hik_left_bound_idx - 1.0;
            h_ik[hik_left_bound_idx] += (k_delta + hik_left_bound - s_ik) / k_delta;
            h_ik[1 + hik_left_bound_idx] += (s_ik - hik_left_bound) / k_delta;
            hik_leftidx_to_cosidx[hik_left_bound_idx].push_back(s_ik_idx);
            hik_rightidx_to_cosidx[1 + hik_left_bound_idx].push_back(s_ik_idx);

            const int s_jk_idx = j < k ? mini_batch_size * j - j * (j + 1) / 2 + k - j - 1 : mini_batch_size * k - k * (k + 1) / 2 + j - k - 1;
            const Dtype s_jk = bottom[0]->cpu_data()[s_jk_idx];
            const int hjk_left_bound_idx = floor((s_jk + 1.0) / k_delta);
            const Dtype hjk_left_bound = k_delta * hjk_left_bound_idx - 1.0;
            h_jk[hjk_left_bound_idx] += (k_delta + hjk_left_bound - s_jk) / k_delta;
            h_jk[1 + hjk_left_bound_idx] += (s_jk - hjk_left_bound) / k_delta;
            hjk_leftidx_to_cosidx[hjk_left_bound_idx].push_back(s_jk_idx);
            hjk_rightidx_to_cosidx[1 + hjk_left_bound_idx].push_back(s_jk_idx);
          }
        }
        caffe_scal<Dtype>(h_size, 1.0 / neg_cnt, &h_ik[0]);
        caffe_scal<Dtype>(h_size, 1.0 / neg_cnt, &h_jk[0]);
        const Dtype hij_left_height = (k_delta + hij_left_bound - s_ij) / k_delta;
        const Dtype hij_right_height = (s_ij - hij_left_bound) / k_delta;
        //LOG(INFO) << "hij_left_height=" << hij_left_height << " hij_right_height=" << hij_right_height;
        for(int k = hij_left_bound_idx; k < h_size; ++k){
          top[0]->mutable_cpu_data()[0] += h_ik[k] * hij_left_height;
          top[0]->mutable_cpu_data()[0] += h_jk[k] * hij_left_height;
          if(k != hij_left_bound_idx){
            top[0]->mutable_cpu_data()[0] += h_ik[k] * hij_right_height;
            top[0]->mutable_cpu_data()[0] += h_jk[k] * hij_right_height;
          }
        }
        //BP
        bottom[0]->mutable_cpu_diff()[s_ij_idx] -= h_ik[hij_left_bound_idx] / k_delta;
        bottom[0]->mutable_cpu_diff()[s_ij_idx] -= h_jk[hij_left_bound_idx] / k_delta;

        for(int cur_neg_idx = hij_left_bound_idx; cur_neg_idx < h_size; ++cur_neg_idx){
          Dtype dL__dh_neg = hij_left_height;
          if(cur_neg_idx != hij_left_bound_idx){
            dL__dh_neg += hij_right_height;
          }
          for(int k = 0; k < hik_leftidx_to_cosidx[cur_neg_idx].size(); ++k){
            bottom[0]->mutable_cpu_diff()[hik_leftidx_to_cosidx[cur_neg_idx][k]] -= dL__dh_neg / k_delta / neg_cnt;
          }
          for(int k = 0; k < hik_rightidx_to_cosidx[cur_neg_idx].size(); ++k){
            bottom[0]->mutable_cpu_diff()[hik_rightidx_to_cosidx[cur_neg_idx][k]] += dL__dh_neg / k_delta / neg_cnt;
          }

          for(int k = 0; k < hjk_leftidx_to_cosidx[cur_neg_idx].size(); ++k){
            bottom[0]->mutable_cpu_diff()[hjk_leftidx_to_cosidx[cur_neg_idx][k]] -= dL__dh_neg / k_delta / neg_cnt;
          }
          for(int k = 0; k < hjk_rightidx_to_cosidx[cur_neg_idx].size(); ++k){
            bottom[0]->mutable_cpu_diff()[hjk_rightidx_to_cosidx[cur_neg_idx][k]] += dL__dh_neg / k_delta / neg_cnt;
          }

        }
      }
    }
  }
  if(pos_cnt != 0){
    top[0]->mutable_cpu_data()[0] /= pos_cnt;
    caffe_scal<Dtype>(bottom[0]->count(), 1.0 / pos_cnt, bottom[0]->mutable_cpu_diff());
  }
}

template <typename Dtype>
void FAPPYLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype k_delta = this->layer_param_.fappy_loss_param().delta();
  const int mini_batch_size = bottom[1]->num();
  caffe_set<Dtype>(top[0]->count(), 0, top[0]->mutable_cpu_data());
  caffe_set<Dtype>(bottom[0]->count(), 0, bottom[0]->mutable_cpu_diff());
  for(double d = 2.0; d >= k_delta; d /= 2.0){
    ElementwiseUpdate(bottom, top, d, mini_batch_size);
    top[0]->mutable_cpu_data()[0] /= 2;
    caffe_scal<Dtype>(bottom[0]->count(), 0.5, bottom[0]->mutable_cpu_diff());
  }
}

template <typename Dtype>
void FAPPYLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if(false == propagate_down[0]) {
    return;
  }
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  caffe_scal<Dtype>(bottom[0]->count(), loss_weight, bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(FAPPYLossLayer);
#endif
INSTANTIATE_CLASS(FAPPYLossLayer);
REGISTER_LAYER_CLASS(FAPPYLoss);
}

