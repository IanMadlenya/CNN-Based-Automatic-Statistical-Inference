#include "caffe/layers/huber_loss_layer.hpp"


#include <math.h>
#include <vector>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
void HuberLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  sign_.ReshapeLike(*bottom[0]);
}

float abs_sum;
const float delta = 10; // 3 is good in case of 7 distributions without cauchy

template <typename Dtype>
void HuberLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  Dtype loss;
  Dtype dot;
  caffe_cpu_sign(count, diff_.cpu_data(), sign_.mutable_cpu_data());
  //abs_sum = caffe_cpu_asum(count, diff_.cpu_data());
  dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  abs_sum = sqrt(dot);
  //std::cout <<  bottom[0]->count() << " " <<  bottom[0]->num() << std::endl;
  if (abs_sum <= delta) {
    loss = dot / bottom[0]->num() / Dtype(2);
  }
  else {
    loss = (delta * abs_sum - delta * delta / Dtype(2)) / bottom[0]->num();
  }
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void HuberLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //int count = bottom[0]->count();
  /*caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());*/

  //caffe_cpu_sign(count, diff_.cpu_data(), sign_.mutable_cpu_data());
  //Dtype abs_sum = caffe_cpu_asum(count, diff_.cpu_data());

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1; // diff_ = bottom[0] - bottom[1]
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      // if-else for huber loss
      // Dtype abs_sum = caffe_cpu_asum(count, top[0].cpu_diff());
      // axpby: Y=alpha * X +beta*Y 
      // bottom[i]->mutable_cpu_diff() = alpha * diff_cpu_data()
      if (abs_sum <= delta) { // L2 loss
          caffe_cpu_axpby(
              bottom[i]->count(),                 // count
              alpha,                              // alpha
              diff_.cpu_data(),                   // a
              Dtype(0),                           // beta
              bottom[i]->mutable_cpu_diff());     // b
      }
      else { // Huber loss
          //const Dtype alpha = sign * delta / bottom[i]->num();
          caffe_cpu_axpby(
              bottom[i]->count(),                 // count
              alpha,                              // alpha
              sign_.cpu_data(),                   // a
              Dtype(0),                           // beta
              bottom[i]->mutable_cpu_diff());     // b
      }
    }
  }
}


INSTANTIATE_CLASS(HuberLossLayer);
REGISTER_LAYER_CLASS(HuberLoss);

}  // namespace caffe
