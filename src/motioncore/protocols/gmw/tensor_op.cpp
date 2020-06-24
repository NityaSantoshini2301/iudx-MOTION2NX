// MIT License
//
// Copyright (c) 2020 Lennart Braun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tensor_op.h"

#include <stdexcept>

#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/linalg_triple_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "gmw_provider.h"
#include "utility/constants.h"
#include "utility/linear_algebra.h"
#include "utility/logger.h"

namespace MOTION::proto::gmw {

template <typename T>
ArithmeticGMWTensorInputSender<T>::ArithmeticGMWTensorInputSender(
    std::size_t gate_id, GMWProvider& gmw_provider, const tensor::TensorDimensions& dimensions,
    ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      input_id_(gmw_provider.get_next_input_id(1)),
      input_future_(std::move(input_future)),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(dimensions)) {
  if (gmw_provider_.get_num_parties() != 2) {
    throw std::logic_error("only two parties are currently supported");
  }
}

template <typename T>
void ArithmeticGMWTensorInputSender<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto my_id = gmw_provider_.get_my_id();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(1 - my_id);
  output_->get_share() = rng.GetUnsigned<T>(input_id_, output_->get_dimensions().get_data_size());

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWTensorInputSender<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto input = input_future_.get();
  if (input.size() != output_->get_dimensions().get_data_size()) {
    throw std::runtime_error("size of input vector != product of expected dimensions");
  }

  // compute my share
  auto& share = output_->get_share();
  std::transform(std::begin(input), std::end(input), std::begin(share), std::begin(share),
                 std::minus{});
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWTensorInputSender::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorInputSender<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorInputReceiver<T>::ArithmeticGMWTensorInputReceiver(
    std::size_t gate_id, GMWProvider& gmw_provider, const tensor::TensorDimensions& dimensions)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      input_id_(gmw_provider.get_next_input_id(1)),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(dimensions)) {}

template <typename T>
void ArithmeticGMWTensorInputReceiver<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorInputReceiver::evaluate_setup start", gate_id_));
    }
  }

  const auto my_id = gmw_provider_.get_my_id();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(1 - my_id);
  output_->get_share() = rng.GetUnsigned<T>(input_id_, output_->get_dimensions().get_data_size());
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorInputReceiver::evaluate_setup end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorInputReceiver<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorOutput<T>::ArithmeticGMWTensorOutput(std::size_t gate_id,
                                                        GMWProvider& gmw_provider,
                                                        ArithmeticGMWTensorCP<T> input,
                                                        std::size_t output_owner)
    : NewGate(gate_id), gmw_provider_(gmw_provider), output_owner_(output_owner), input_(input) {
  auto my_id = gmw_provider_.get_my_id();
  if (output_owner_ == my_id) {
    share_future_ = gmw_provider_.register_for_ints_message<T>(
        1 - my_id, gate_id_, input_->get_dimensions().get_data_size());
  }
}

template <typename T>
void ArithmeticGMWTensorOutput<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorOutput::evaluate_online start", gate_id_));
    }
  }

  auto my_id = gmw_provider_.get_my_id();
  if (output_owner_ == my_id) {
    auto other_share = share_future_.get();
    assert(other_share.size() == input_->get_dimensions().get_data_size());
    std::transform(std::begin(other_share), std::end(other_share), std::begin(input_->get_share()),
                   std::begin(other_share), std::plus{});
    output_promise_.set_value(std::move(other_share));
  } else {
    gmw_provider_.send_ints_message(1 - my_id, gate_id_, input_->get_share());
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorOutput::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<std::vector<T>> ArithmeticGMWTensorOutput<T>::get_output_future() {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

template class ArithmeticGMWTensorOutput<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorConv2D<T>::ArithmeticGMWTensorConv2D(std::size_t gate_id,
                                                        GMWProvider& gmw_provider,
                                                        tensor::Conv2DOp conv_op,
                                                        const ArithmeticGMWTensorCP<T> input,
                                                        const ArithmeticGMWTensorCP<T> kernel,
                                                        const ArithmeticGMWTensorCP<T> bias)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      conv_op_(conv_op),
      input_(input),
      kernel_(kernel),
      bias_(bias),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(conv_op.get_output_tensor_dims())),
      triple_index_(
          gmw_provider.get_linalg_triple_provider().register_for_conv2d_triple<T>(conv_op)),
      share_future_(gmw_provider_.register_for_ints_message<T>(
          1 - gmw_provider.get_my_id(), this->gate_id_,
          conv_op.compute_input_size() + conv_op.compute_kernel_size())) {}

template <typename T>
void ArithmeticGMWTensorConv2D<T>::evaluate_online() {
  auto& ltp = gmw_provider_.get_linalg_triple_provider();
  auto triple = ltp.get_conv2d_triple<T>(conv_op_, triple_index_);
  this->input_->wait_online();
  this->kernel_->wait_online();
  const auto& input_buffer = this->input_->get_share();
  const auto& kernel_buffer = this->kernel_->get_share();
  const auto input_size = conv_op_.compute_input_size();
  const auto kernel_size = conv_op_.compute_kernel_size();
  assert(input_buffer.size() == input_size);
  assert(kernel_buffer.size() == kernel_size);

  const auto my_id = gmw_provider_.get_my_id();

  //  mask inputs
  std::vector<T> de(input_size + kernel_size);
  auto it = std::transform(std::begin(input_buffer), std::end(input_buffer), std::begin(triple.a_),
                           std::begin(de), std::minus{});
  std::transform(std::begin(kernel_buffer), std::end(kernel_buffer), std::begin(triple.b_), it,
                 std::minus{});
  this->gmw_provider_.send_ints_message(1 - my_id, this->gate_id_, de);

  // compute d, e
  auto other_share = share_future_.get();
  std::transform(std::begin(de), std::end(de), std::begin(other_share), std::begin(de),
                 std::plus{});

  // result = c ...
  std::vector<T> result(std::move(triple.c_));
  std::vector<T> tmp(result.size());
  // ... - d * e ...
  if (this->gmw_provider_.is_my_job(this->gate_id_)) {
    convolution(conv_op_, de.data(), de.data() + input_size, tmp.data());
    std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                   std::minus{});
  }
  // ... + e * x + d * y
  convolution(conv_op_, input_buffer.data(), de.data() + input_size, tmp.data());
  std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                 std::plus{});
  convolution(conv_op_, de.data(), kernel_buffer.data(), tmp.data());
  std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                 std::plus{});
  this->output_->get_share() = std::move(result);
  this->output_->set_online_ready();
}

template class ArithmeticGMWTensorConv2D<std::uint64_t>;

}  // namespace MOTION::proto::gmw