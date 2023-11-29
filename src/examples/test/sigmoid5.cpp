/*
This code reads shares from build_debwithrelinfo/ShareFiles/A.txt and
build_debwithrelinfo/ShareFiles/B.txt and performs a ReLU operation on them.
If this code is run with a function to write output shares (in tensor_op.cpp)
output shares of this will be written. The following instructions run this code.

At the argument "--filepath " give the path of the file containing shares from build_deb.... folder
Server-0
./bin/sigmoid5 --my-id 0 --party 0,::1,7002 --party 1,::1,7000 --arithmetic-protocol beavy
--boolean-protocol yao --fractional-bits 13 --file-input outputshare_0

Server-1
./bin/sigmoid5 --my-id 1 --party 0,::1,7002 --party 1,::1,7000 --arithmetic-protocol beavy
--boolean-protocol yao --fractional-bits 13  --file-input outputshare_1

*/
// MIT License
//
// Copyright (c) 2021 Lennart Braun
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <stdexcept>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <boost/json/serialize.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include "algorithm/circuit_loader.h"
#include "base/gate_factory.h"
#include "base/two_party_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "compute_server/compute_server.h"
#include "statistics/analysis.h"
#include "utility/logger.h"

#include "base/two_party_tensor_backend.h"
#include "protocols/beavy/tensor.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"
#include "utility/new_fixed_point.h"

namespace po = boost::program_options;
int j = 0;

static std::vector<uint64_t> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
  return MOTION::Helpers::RandomVector<uint64_t>(dims.get_data_size());
}

struct Matrix {
  std::vector<uint64_t> Delta;
  std::vector<uint64_t> delta;
  int row;
  int col;
};

struct Options {
  std::size_t threads;
  bool json;
  std::size_t num_repetitions;
  std::size_t num_simd;
  bool sync_between_setup_and_online;
  MOTION::MPCProtocol arithmetic_protocol;
  MOTION::MPCProtocol boolean_protocol;
  //////////////////////////changes////////////////////////////
  int num_elements;
  //////////////////////////////////////////////////////////////
  std::size_t fractional_bits;
  std::string inputpath;

  std::size_t my_id;
  // std::string filepath_frombuild;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
  Matrix input;
  Matrix row;
  Matrix col;
};

//////////////////New functions////////////////////////////////////////
/// In read_file also include file not there error and file empty alerts
std::uint64_t read_file(std::ifstream& pro) {
  std::string str;
  char num;
  while (pro >> std::noskipws >> num) {
    if (num != ' ' && num != '\n') {
      str.push_back(num);
    } else {
      break;
    }
  }

  std::string::size_type sz = 0;
  std::uint64_t ret = (uint64_t)std::stoull(str, &sz, 0);
  return ret;
}

void read_input(Options* options, std::string p) {
  std::ifstream temps;
  temps.open(p);
  std::cout << "p:" << p << "\n";
  if (temps) {
    std::cout << "File found\n";
  } else {
    std::cout << "File not found\n";
  }
  assert(temps);

  std::uint64_t rows = read_file(temps);
  options->input.row = rows;
  std::cout << "r " << rows << " ";
  std::uint64_t cols = read_file(temps);
  options->input.col = cols;
  std::cout << "c " << cols << "\n";

  for (int i = 0; i < rows * cols; ++i) {
    uint64_t m1 = read_file(temps);
    options->input.Delta.push_back(m1);
    uint64_t m2 = read_file(temps);
    options->input.delta.push_back(m2);
  }
  temps.close();
}

void file_read(Options* options) {
  std::string path = std::filesystem::current_path();
  // std::string t1 = path + "/" + options->inputpath;
  std::string t1 = path + "/server" + std::to_string(options->my_id) + "/" + options->inputpath;
  std::cout << t1 << "\n";
  read_input(options, t1);
}

//////////////////////////////////////////////////////////////////////
std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("file-input", po::value<std::string>()->required(), "config file containing options")
    ("my-id", po::value<std::size_t>()->required(), "my party id")
    ("party", po::value<std::vector<std::string>>()->multitoken(),
     "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("threads", po::value<std::size_t>()->default_value(0), "number of threads to use for gate evaluation")
    ("json", po::bool_switch()->default_value(false), "output data in JSON format")
    ("fractional-bits", po::value<std::size_t>()->default_value(16),
     "number of fractional bits for fixed-point arithmetic")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("sync-between-setup-and-online", po::bool_switch()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false), "just build the circuit, but not execute it")
    ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  bool help = vm["help"].as<bool>();
  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }
  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::store(po::parse_config_file(ifs, desc), vm);
  }
  try {
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error:" << e.what() << "\n\n";
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  options.my_id = vm["my-id"].as<std::size_t>();
  options.threads = vm["threads"].as<std::size_t>();
  options.json = vm["json"].as<bool>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.num_simd = vm["num-simd"].as<std::size_t>();
  options.sync_between_setup_and_online = vm["sync-between-setup-and-online"].as<bool>();
  options.no_run = vm["no-run"].as<bool>();
  //////////////////////////////////////////////////////////////////
  options.inputpath = vm["file-input"].as<std::string>();
  /////////////////////////////////////////////////////////////////
  options.fractional_bits = vm["fractional-bits"].as<std::size_t>();
  if (options.my_id > 1) {
    std::cerr << "my-id must be one of 0 and 1\n";
    return std::nullopt;
  }

  auto arithmetic_protocol = vm["arithmetic-protocol"].as<std::string>();
  boost::algorithm::to_lower(arithmetic_protocol);
  if (arithmetic_protocol == "gmw") {
    options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticGMW;
  } else if (arithmetic_protocol == "beavy") {
    options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticBEAVY;
  } else {
    std::cerr << "invalid protocol: " << arithmetic_protocol << "\n";
    return std::nullopt;
  }
  auto boolean_protocol = vm["boolean-protocol"].as<std::string>();
  boost::algorithm::to_lower(boolean_protocol);
  if (boolean_protocol == "yao") {
    options.boolean_protocol = MOTION::MPCProtocol::Yao;
  } else if (boolean_protocol == "gmw") {
    options.boolean_protocol = MOTION::MPCProtocol::BooleanGMW;
  } else if (boolean_protocol == "beavy") {
    options.boolean_protocol = MOTION::MPCProtocol::BooleanBEAVY;
  } else {
    std::cerr << "invalid protocol: " << boolean_protocol << "\n";
    return std::nullopt;
  }

  //////////////////////////////////////////////////////////////////
  file_read(&options);
  ////////////////////////////////////////////////////////////////////

  const auto parse_party_argument =
      [](const auto& s) -> std::pair<std::size_t, MOTION::Communication::tcp_connection_config> {
    const static std::regex party_argument_re("([01]),([^,]+),(\\d{1,5})");
    std::smatch match;
    if (!std::regex_match(s, match, party_argument_re)) {
      throw std::invalid_argument("invalid party argument");
    }
    auto id = boost::lexical_cast<std::size_t>(match[1]);
    auto host = match[2];
    auto port = boost::lexical_cast<std::uint16_t>(match[3]);
    return {id, {host, port}};
  };

  const std::vector<std::string> party_infos = vm["party"].as<std::vector<std::string>>();
  if (party_infos.size() != 2) {
    std::cerr << "expecting two --party options\n";
    return std::nullopt;
  }

  options.tcp_config.resize(2);
  std::size_t other_id = 2;

  const auto [id0, conn_info0] = parse_party_argument(party_infos[0]);
  const auto [id1, conn_info1] = parse_party_argument(party_infos[1]);
  if (id0 == id1) {
    std::cerr << "need party arguments for party 0 and 1\n";
    return std::nullopt;
  }
  options.tcp_config[id0] = conn_info0;
  options.tcp_config[id1] = conn_info1;

  return options;
}

std::unique_ptr<MOTION::Communication::CommunicationLayer> setup_communication(
    const Options& options) {
  MOTION::Communication::TCPSetupHelper helper(options.my_id, options.tcp_config);
  return std::make_unique<MOTION::Communication::CommunicationLayer>(options.my_id,
                                                                     helper.setup_connections());
}

void print_stats(const Options& options,
                 const MOTION::Statistics::AccumulatedRunTimeStats& run_time_stats,
                 const MOTION::Statistics::AccumulatedCommunicationStats& comm_stats) {
  if (options.json) {
    auto obj = MOTION::Statistics::to_json("const add", run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("arithmetic_protocol", MOTION::ToString(options.arithmetic_protocol));
    obj.emplace("boolean_protocol", MOTION::ToString(options.boolean_protocol));
    obj.emplace("simd", options.num_simd);
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    std::cout << obj << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats("sigmoid", run_time_stats, comm_stats);
  }
}

auto create_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  // retrieve the gate factories for the chosen protocols
  auto& arithmetic_tof = backend.get_tensor_op_factory(options.arithmetic_protocol);
  auto& boolean_tof = backend.get_tensor_op_factory(MOTION::MPCProtocol::Yao);

  std::cout << options.input.row << " " << options.input.col << "\n";

  MOTION::tensor::TensorDimensions X_dims = {.batch_size_ = 1,
                                             .num_channels_ = 1,
                                             .height_ = options.input.row,
                                             .width_ = options.input.col};

  MOTION::tensor::TensorCP tensor_X;

  auto pairX = arithmetic_tof.make_arithmetic_64_tensor_input_shares(X_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_X =
      std::move(pairX.first);
  tensor_X = pairX.second;

  input_promises_X[0].set_value(options.input.Delta);
  input_promises_X[1].set_value(options.input.delta);

  std::function<MOTION::tensor::TensorCP(const MOTION::tensor::TensorCP&)> make_activation,
      make_relu;
  std::function<MOTION::tensor::TensorCP(const MOTION::tensor::TensorCP&, std::size_t)>
      make_indicator, make_sigmoid, make_sigmoid5;

  // -RELU(-X)
  make_activation = [&](const auto& input) {
    //  const auto negated_tensor = arithmetic_tof.make_tensor_negate(input);
    const auto boolean_tensor = boolean_tof.make_tensor_conversion(MOTION::MPCProtocol::Yao, input);
    const auto relu_tensor = boolean_tof.make_tensor_relu_op(boolean_tensor);
    return boolean_tof.make_tensor_conversion(options.arithmetic_protocol, relu_tensor);
  };

  //  RELU(X)
  make_relu = [&](const auto& input) {
    const auto negated_tensor = arithmetic_tof.make_tensor_negate(input);
    const auto boolean_tensor =
        boolean_tof.make_tensor_conversion(MOTION::MPCProtocol::Yao, negated_tensor);
    const auto relu_tensor = boolean_tof.make_tensor_relu_op(boolean_tensor);  // -RELU(-X)
    const auto finBoolean_tensor =
        boolean_tof.make_tensor_conversion(options.arithmetic_protocol, relu_tensor);
    return arithmetic_tof.make_tensor_negate(finBoolean_tensor);
  };

  make_sigmoid = [&](const auto& input, std::size_t input_size) {
    const std::vector<uint64_t> constant_vector1(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(-0.5, options.fractional_bits));
    const auto input_const_add = arithmetic_tof.make_tensor_constAdd_op(input, constant_vector1);
    const auto first_relu_output = make_activation(input_const_add);
    const std::vector<uint64_t> constant_vector2(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(1, options.fractional_bits));
    const auto input_const_add2 =
        arithmetic_tof.make_tensor_constAdd_op(first_relu_output, constant_vector2);
    const auto negated_tensor = arithmetic_tof.make_tensor_negate(input_const_add2);
    const auto final_relu_output = make_activation(negated_tensor);
    return arithmetic_tof.make_tensor_negate(final_relu_output);
  };

  make_indicator = [&](const auto& input, std::size_t input_size) {
    const auto first_relu_output = make_activation(input);  // Returns -RELU(-X)

    // Declaring a constant uint64 vector of same size as input and initializing every element with
    // encoded 9000
    std::vector<uint64_t> const_vector(input_size, MOTION::new_fixed_point::encode<uint64_t, float>(
                                                       9000, options.fractional_bits));

    // Multiplying the tensor with the constant vector (element wise)
    const auto mult_output = arithmetic_tof.make_tensor_constMul_op(first_relu_output, const_vector,
                                                                    options.fractional_bits);
    // Reached 9000 * -RELU(-X)
    // Adding an encoded one to the tensor
    std::vector<uint64_t> const_vector2(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(1, options.fractional_bits));
    const auto add_output = arithmetic_tof.make_tensor_constAdd_op(mult_output, const_vector2);
    // Reached 1 + 9000 * -RELU(-X)

    return make_relu(add_output);  // make_relu returns RELU(Y)
    // Returning RELU( 1 + 9000 * -RELU(-X) )
  };

  /*
10 ^-4, x ≤ -5
0.02776 · x + 0.145, -5 < x ≤ -2.5
0.17 · x + 0.5, -2.5 < x ≤ 2.5
0.02776 · x + 0.85498, 2.5 < x ≤ 5
1 - 10^4, x > 5
*/
  make_sigmoid5 = [&](const auto& input, std::size_t input_size) {
    const std::vector<uint64_t> vector_a(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(-5, options.fractional_bits));
    const std::vector<uint64_t> vector_b(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(-2.5, options.fractional_bits));
    const std::vector<uint64_t> vector_c(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(2.5, options.fractional_bits));
    const std::vector<uint64_t> vector_d(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(5, options.fractional_bits));

    const std::vector<uint64_t> constant_vector1(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.0001, options.fractional_bits));
    const std::vector<uint64_t> constant_vector2(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.02776, options.fractional_bits));
    const std::vector<uint64_t> constant_vector3(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.145, options.fractional_bits));
    const std::vector<uint64_t> constant_vector4(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.17, options.fractional_bits));
    const std::vector<uint64_t> constant_vector5(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(0.5, options.fractional_bits));
    const std::vector<uint64_t> constant_vector6(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.02776, options.fractional_bits));
    const std::vector<uint64_t> constant_vector7(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.85498, options.fractional_bits));
    const std::vector<uint64_t> constant_vector8(
        input_size,
        MOTION::new_fixed_point::encode<uint64_t, float>(0.9999, options.fractional_bits));

    const std::vector<uint64_t> vector_one(
        input_size, MOTION::new_fixed_point::encode<uint64_t, float>(1, options.fractional_bits));

    // I(x + 5 >= 0)
    const auto indicator1 =
        make_indicator(arithmetic_tof.make_tensor_constAdd_op(input, vector_d), input_size);
    const auto negated_indicator1 = arithmetic_tof.make_tensor_negate(indicator1);
    // 1 - I(x + 5 >= 0)
    const auto indicator2 = arithmetic_tof.make_tensor_constAdd_op(negated_indicator1, vector_one);
    // I(x + 2.5 >= 0)
    const auto indicator3 =
        make_indicator(arithmetic_tof.make_tensor_constAdd_op(input, vector_c), input_size);
    const auto negated_indicator3 = arithmetic_tof.make_tensor_negate(indicator3);
    // 1 - I(x + 2.5 >= 0)
    const auto indicator4 = arithmetic_tof.make_tensor_constAdd_op(negated_indicator3, vector_one);
    // I(x - 2.5 >= 0)
    const auto indicator5 =
        make_indicator(arithmetic_tof.make_tensor_constAdd_op(input, vector_b), input_size);
    const auto negated_indicator5 = arithmetic_tof.make_tensor_negate(indicator5);
    // 1 - I(x - 2.5 >= 0)
    const auto indicator6 = arithmetic_tof.make_tensor_constAdd_op(negated_indicator5, vector_one);
    // I(x - 5 >= 0)
    const auto indicator7 =
        make_indicator(arithmetic_tof.make_tensor_constAdd_op(input, vector_a), input_size);
    const auto negated_indicator7 = arithmetic_tof.make_tensor_negate(indicator7);
    // 1 - I(x - 5 >= 0)
    const auto indicator8 = arithmetic_tof.make_tensor_constAdd_op(negated_indicator7, vector_one);

    const MOTION::tensor::HammOp hamm_op = {
        .input_A_shape_ = {options.input.row, options.input.col},
        .input_B_shape_ = {options.input.row, options.input.col},
        .output_shape_ = {options.input.row, options.input.col}};

    MOTION::tensor::TensorCP indicator_hamm_output1 = arithmetic_tof.make_tensor_hamm_op(
        hamm_op, indicator1, indicator4, options.fractional_bits);
    MOTION::tensor::TensorCP indicator_hamm_output2 = arithmetic_tof.make_tensor_hamm_op(
        hamm_op, indicator3, indicator6, options.fractional_bits);
    MOTION::tensor::TensorCP indicator_hamm_output3 = arithmetic_tof.make_tensor_hamm_op(
        hamm_op, indicator5, indicator8, options.fractional_bits);

    const auto piecewise2 = arithmetic_tof.make_tensor_constAdd_op(
        arithmetic_tof.make_tensor_constMul_op(input, constant_vector2, options.fractional_bits),
        constant_vector3);
    const auto piecewise3 = arithmetic_tof.make_tensor_constAdd_op(
        arithmetic_tof.make_tensor_constMul_op(input, constant_vector4, options.fractional_bits),
        constant_vector5);
    const auto piecewise4 = arithmetic_tof.make_tensor_constAdd_op(
        arithmetic_tof.make_tensor_constMul_op(input, constant_vector6, options.fractional_bits),
        constant_vector7);

    const auto output1 = arithmetic_tof.make_tensor_constMul_op(indicator2, constant_vector1,
                                                                options.fractional_bits);
    MOTION::tensor::TensorCP output2 = arithmetic_tof.make_tensor_hamm_op(
        hamm_op, piecewise2, indicator_hamm_output1, options.fractional_bits);
    MOTION::tensor::TensorCP output3 = arithmetic_tof.make_tensor_hamm_op(
        hamm_op, piecewise3, indicator_hamm_output2, options.fractional_bits);
    MOTION::tensor::TensorCP output4 = arithmetic_tof.make_tensor_hamm_op(
        hamm_op, piecewise4, indicator_hamm_output3, options.fractional_bits);
    const auto output5 = arithmetic_tof.make_tensor_constMul_op(indicator7, constant_vector8,
                                                                options.fractional_bits);

    const auto add_output1 = arithmetic_tof.make_tensor_add_op(output1, output2);
    const auto add_output2 = arithmetic_tof.make_tensor_add_op(add_output1, output3);
    const auto add_output3 = arithmetic_tof.make_tensor_add_op(add_output2, output4);
    const auto final_output = arithmetic_tof.make_tensor_add_op(add_output3, output5);

    return final_output;
  };

  // 7 -5 -4 -2.5 -2 0 2 2.5 4 5 7
  //-0.00012207 , 0.00610352 , 0.0338135 , 0.0749512 , 0.159912 , 0.499756 , 0.8396 , 0.923828 ,
  // 0.965454 , 0.999756 , 0.999756

  MOTION::tensor::TensorCP tensor_sigmoid = make_sigmoid5(tensor_X, X_dims.get_data_size());

  ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> output_future, main_output_future,
      main_output;

  if (options.my_id == 0) {
    arithmetic_tof.make_arithmetic_tensor_output_other(tensor_sigmoid);
  } else {
    main_output_future = arithmetic_tof.make_arithmetic_64_tensor_output_my(tensor_sigmoid);
  }

  return std::move(main_output_future);
}

void run_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto output_future = create_composite_circuit(options, backend);
  backend.run();
  if (options.my_id == 1) {
    auto main = output_future.get();

    for (int i = 0; i < main.size(); ++i) {
      long double temp =
          MOTION::new_fixed_point::decode<uint64_t, long double>(main[i], options.fractional_bits);

      std::cout << temp << " , ";
    }
  }
}

int main(int argc, char* argv[]) {
  std::cout << "Inside main";
  auto options = parse_program_options(argc, argv);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  try {
    auto comm_layer = setup_communication(*options);
    auto logger = std::make_shared<MOTION::Logger>(options->my_id,
                                                   boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);
    MOTION::Statistics::AccumulatedRunTimeStats run_time_stats;
    MOTION::Statistics::AccumulatedCommunicationStats comm_stats;
    MOTION::TwoPartyTensorBackend backend(*comm_layer, options->threads,
                                          options->sync_between_setup_and_online, logger);
    run_composite_circuit(*options, backend);
    comm_layer->sync();
    comm_stats.add(comm_layer->get_transport_statistics());
    comm_layer->reset_transport_statistics();
    run_time_stats.add(backend.get_run_time_stats());
    comm_layer->shutdown();
    print_stats(*options, run_time_stats, comm_stats);
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}