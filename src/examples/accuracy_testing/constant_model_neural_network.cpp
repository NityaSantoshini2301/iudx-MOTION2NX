/*
inputpath changed to imageprovider
imageprovider is file name inside server 0 and server 1 respectively
eg. (ip1_0 , ip1_1) , (ip2_0,ip2_1) and so on..
from bash file we are giving ip1 and in this file it is appended to ip1_0 and ip1_1

At the argument "--filepath " give the path of the file containing shares from build_deb.... folder
Server-0
./bin/constant_model_neural_network --my-id 0 --party 0,::1,7002 --party 1,::1,7000
--fractional-bits 13 --layer-id 1 --current-path ${BASE_DIR}/build_debwithrelinfo_gcc
--config-file-input remote_image_shares --weights-file newW1 --bias-file newB1 --boolean-protocol
yao --arithmetic-protocol beavy


Server-1
./bin/constant_model_neural_network --my-id 1 --party 0,::1,7002 --party 1,::1,7000
--fractional-bits 13 --layer-id 1 --current-path ${BASE_DIR}/build_debwithrelinfo_gcc
--config-file-input remote_image_shares --weights-file newW1 --bias-file newB1 --boolean-protocol
yao --arithmetic-protocol beavy

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
#include <parallel/algorithm>

#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "compute_server/compute_server.h"
#include "statistics/analysis.h"
#include "utility/logger.h"

#include <boost/thread/thread.hpp>
#include <chrono>
#include "algorithm/circuit_loader.h"
#include "base/gate_factory.h"
#include "base/two_party_backend.h"
#include "base/two_party_tensor_backend.h"
#include "communication/message_handler.h"
#include "protocols/beavy/tensor.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"
#include "utility/linear_algebra.h"
#include "utility/new_fixed_point.h"

namespace po = boost::program_options;

int j = 0;
// global variable
std::vector<std::uint64_t> Delta_otherid;
std::vector<std::uint64_t> delta_1;
bool flag1, flag2, flag0 = false;
bool server1_ready_flag, server0_ready_flag = false;

static std::vector<uint64_t> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
  return MOTION::Helpers::RandomVector<uint64_t>(dims.get_data_size());
}

bool is_empty(std::ifstream& file) { return file.peek() == std::ifstream::traits_type::eof(); }

void testMemoryOccupied(int WriteToFiles, int my_id, std::string path) {
  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();

  long page_size_kb =
      sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;
  std::cout << "RSS - " << rss << " kB\n";
  double shared_mem = share * page_size_kb;
  std::cout << "Shared Memory - " << shared_mem << " kB\n";
  std::cout << "Private Memory - " << rss - shared_mem << "kB\n";
  std::cout << std::endl;
  if (WriteToFiles == 1) {
    /////// Generate path for the AverageMemoryDetails file and MemoryDetails file
    std::string t1 = path + "/" + "AverageMemoryDetails" + std::to_string(my_id);
    std::string t2 = path + "/" + "MemoryDetails" + std::to_string(my_id);

    ///// Write to the AverageMemoryDetails files
    std::ofstream file1;
    file1.open(t1, std::ios_base::app);
    file1 << rss;
    file1 << "\n";
    file1.close();

    std::ofstream file2;
    file2.open(t2, std::ios_base::app);
    file2 << "Multiplication layer : \n";
    file2 << "RSS - " << rss << " kB\n";
    file2 << "Shared Memory - " << shared_mem << " kB\n";
    file2 << "Private Memory - " << rss - shared_mem << "kB\n";
    file2.close();
  }
}

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
  std::size_t fractional_bits;
  std::string imageprovider;
  std::string modelpath;
  std::size_t layer_id;
  std::string currentpath;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
  Matrix image_file;
  Matrix row;
  Matrix col;
  std::vector<float> B_data;
  std::vector<float> W_data;
  std::uint64_t W_row = 0, W_col = 0;
  std::uint64_t B_row = 0, B_col = 0;
  std::string weights_file;
  std::string bias_file;
  MOTION::MPCProtocol arithmetic_protocol;
  MOTION::MPCProtocol boolean_protocol;
};

// this function reads all lines but takes into consideration only the required input
// if j==count , the program will not make str empty and return it once j>count
std::string read_filepath(std::ifstream& indata, int count) {
  std::string str;

  while (j <= count) {
    char num;
    while (indata) {
      std::getline(indata, str);

      if (j < count) {
        str = " ";
      }
      j++;
      break;
    }
  }
  return str;
}

int image_shares(Options* options, std::string p) {
  std::ifstream temps;
  try {
    temps.open(p);
    if (temps) {
      std::cout << "Image shares File found\n";
    } else {
      std::cout << "Image shares File not found\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening the image share file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(temps)) {
      // file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Image share input file is empty.\n";
    return EXIT_FAILURE;
  }

  std::uint64_t rows, cols;
  try {
    rows = read_file(temps);
    options->image_file.row = rows;
    std::cout << "Image r " << rows << " ";
    cols = read_file(temps);
    options->image_file.col = cols;
    std::cout << "c " << cols << "\n";
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while reading columns from image shares.\n";
    return EXIT_FAILURE;
  }

  for (int i = 0; i < rows * cols; ++i) {
    try {
      uint64_t m1 = read_file(temps);
      options->image_file.Delta.push_back(m1);
      uint64_t m2 = read_file(temps);
      options->image_file.delta.push_back(m2);
      // std::cout << options->image_file.Delta[i] << " " << options->image_file.delta[i] << "\n";
    } catch (std::ifstream::failure e) {
      std::cerr << "Error while reading columns from image shares.\n";
      return EXIT_FAILURE;
    }
  }
  temps.close();
}

int read_W_csv(Options* options, std::string W_path) {
  std::ifstream file;
  std::cout << "W: " << W_path << "\n";
  try {
    file.open(W_path);
    if (file) {
      std::cout << "W_csv File found\n";
    } else {
      std::cout << "W_csv File not found\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening the weight csv file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(file)) {
      // //file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "W_csv file is empty.\n";
    return EXIT_FAILURE;
  }

  // //read weights
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      options->W_data.push_back(stof(cell));
      //   std::cout << stof(cell) << " ";
    }
  }
  //   std::cout << "\n";

  file.close();
  file.open(W_path);

  while (std::getline(file, line)) {
    float temp;
    std::istringstream iss(line);
    if (iss >> temp) {
      options->W_row++;
    }
  }
  int total_size = options->W_data.size();
  options->W_col = total_size / (options->W_row);

  std::cout << "W rows : " << options->W_row << "\n";
  std::cout << "W col : " << options->W_col << "\n";
  file.close();
}

int read_B_csv(Options* options, std::string B_path) {
  std::ifstream file;
  std::cout << "B: " << B_path << "\n";
  try {
    file.open(B_path);
    if (file) {
      std::cout << "B_csv File found\n";
    } else {
      std::cout << "B_csv File not found\n";
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "Error while opening the Bias csv file.\n";
    return EXIT_FAILURE;
  }

  try {
    if (is_empty(file)) {
      // //file is empty
    }
  } catch (std::ifstream::failure e) {
    std::cerr << "B_csv file is empty.\n";
    return EXIT_FAILURE;
  }

  // //read bias
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      options->B_data.push_back(stof(cell));
      std::cout << stof(cell) << " ";
    }
  }
  std::cout << "\n";

  file.close();
  file.open(B_path);

  while (std::getline(file, line)) {
    float temp;
    std::istringstream iss(line);
    if (iss >> temp) {
      options->B_row++;
    }
  }
  std::cout << "B rows:" << options->B_row << "\n";
  options->B_col = 1;

  file.close();
}

int file_read(Options* options) {
  std::string path = options->currentpath;
  std::string t1, i;
  // //if layer_id=1 then read filename inside server
  // //else read file_config_input (id is appended)
  if (options->layer_id == 1) {
    t1 = path + "/server" + std::to_string(options->my_id) + "/Image_shares/" +
         options->imageprovider;
  } else if (options->layer_id > 1) {
    // outputshare_0/1 inside server 1/0
    t1 = path + "/server" + std::to_string(options->my_id) + "/" + options->imageprovider + "_" +
         std::to_string(options->my_id);
  }

  image_shares(options, t1);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::string home_dir = getenv("BASE_DIR");
  std::string model_path = home_dir + "/data/ModelProvider/";
  std::string W1_path, W2_path, B1_path, B2_path;
  std::cout << model_path << "\n";

  W1_path = model_path + options->weights_file + ".csv";
  B1_path = model_path + options->bias_file + ".csv";

  read_W_csv(options, W1_path);
  read_B_csv(options, B1_path);
}

//////////////////////////////////////////////////////////////////////
std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("config-file-input", po::value<std::string>()->required(), "config file containing options")
    ("my-id", po::value<std::size_t>()->required(), "my party id")
    ("layer-id", po::value<std::size_t>()->required(), "layer id")
    ("party", po::value<std::vector<std::string>>()->multitoken(),
     "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("threads", po::value<std::size_t>()->default_value(0), "number of threads to use for gate evaluation")
    ("json", po::bool_switch()->default_value(false), "output data in JSON format")
    ("fractional-bits", po::value<std::size_t>()->default_value(16),
     "number of fractional bits for fixed-point arithmetic")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("current-path",po::value<std::string>()->required(), "current path build_debwithrelinfo")
    ("sync-between-setup-and-online", po::bool_switch()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false), "just build the circuit, but not execute it")
    ("weights-file", po::value<std::string>()->required(), "weights csv file")
    ("bias-file", po::value<std::string>()->required(), "biass csv file")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ;
  // //clang-format on

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
  options.currentpath = vm["current-path"].as<std::string>();
  //////////////////////////////////////////////////////////////////
  options.imageprovider = vm["config-file-input"].as<std::string>();
  options.layer_id = vm["layer-id"].as<std::size_t>();
  options.weights_file = vm["weights-file"].as<std::string>();
  options.bias_file = vm["bias-file"].as<std::string>();
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
    auto obj = MOTION::Statistics::to_json("tensor_gt_mul", run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("simd", options.num_simd);
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    // std::cout << obj << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats("read_image_shares", run_time_stats, comm_stats);
  }
}

auto create_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  std::cout << "Inside create_composite"
            << "\n";

  auto frac_bits=options.fractional_bits;
  // retrieve the gate factories for the chosen protocols
  auto& arithmetic_tof = backend.get_tensor_op_factory(options.arithmetic_protocol);
  auto& boolean_tof = backend.get_tensor_op_factory(MOTION::MPCProtocol::Yao);

  const MOTION::tensor::GemmOp gemm_op1 = {
      .input_A_shape_ = {options.W_row, options.W_col},
      .input_B_shape_ = {options.image_file.row, options.image_file.col},
      .output_shape_ = {options.W_row, options.image_file.col}};

 const auto X_dims = gemm_op1.get_input_B_tensor_dims();
    //  const auto X_dims = options.image_file.row;

  /////////////////////////////////////////////////////////////////////////
  
  MOTION::tensor::TensorCP tensor_X ,cmm_output,add_output;

  auto pairX = arithmetic_tof.make_arithmetic_64_tensor_input_shares(X_dims);
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises_X =
      std::move(pairX.first);
  tensor_X = pairX.second;
  
  ///////////////////////////////////////////////////////////////
  input_promises_X[0].set_value(options.image_file.Delta);
  input_promises_X[1].set_value(options.image_file.delta);

  ///////////////////////////////////////////////////////////////////

  std::vector<std::uint64_t> encoded_W(options.W_data.size(), 0);
  std::transform(options.W_data.begin(),options.W_data.end(), encoded_W.begin(), [frac_bits](auto j) {
    return MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
  });

    std::vector<std::uint64_t> encoded_B(options.B_data.size(), 0);
  std::transform(options.B_data.begin(),options.B_data.end(), encoded_B.begin(), [frac_bits](auto j) {
    return MOTION::new_fixed_point::encode<std::uint64_t, float>(j, frac_bits);
  });
   

     // In this function call, when 'true' or 'false' is set as the fourth argument,
  // 'true' multiplies tensor * constant, while 'false' does constant * tensor.
   cmm_output = arithmetic_tof.make_tensor_constMatrix_Mul_op(gemm_op1, encoded_W, tensor_X,false, options.fractional_bits);
   
   add_output = arithmetic_tof.make_tensor_constAdd_op(cmm_output,encoded_B);

  ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> output_future, main_output_future,
      main_output;

  if (options.my_id == 0) {
    arithmetic_tof.make_arithmetic_tensor_output_other(add_output);
  } else {
    main_output_future = arithmetic_tof.make_arithmetic_64_tensor_output_my(add_output);
  }

  return std::move(main_output_future);
}

void run_composite_circuit(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto output_future = create_composite_circuit(options, backend);
  backend.run();
  if (options.my_id == 1) {
    auto interm = output_future.get();
    std::cout<<"The output is : \n";

    for (int i = 0; i < interm.size(); ++i) {
      long double temp =
          MOTION::new_fixed_point::decode<uint64_t, long double>(interm[i], options.fractional_bits);
      //     mod_x.push_back(temp);
      std::cout << temp << ",";

    }
  }
}

int main(int argc, char* argv[]) 
{
  auto options = parse_program_options(argc, argv);
  int WriteToFiles = 1;
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
    testMemoryOccupied(WriteToFiles, options->my_id, options->currentpath);
    comm_layer->shutdown();
    print_stats(*options, run_time_stats, comm_stats);
    if (WriteToFiles == 1) {
      /////// Generate path for the AverageTimeDetails file and MemoryDetails file
      // std::string path = std::filesystem::current_path();
      std::string path = options->currentpath;
      std::string t1 = path + "/" + "AverageTimeDetails" + std::to_string(options->my_id);
      std::string t2 = path + "/" + "MemoryDetails" + std::to_string(options->my_id);

      ///// Write to the AverageMemoryDetails files
      std::ofstream file2;
      file2.open(t2, std::ios_base::app);
      std::string time_str =
          MOTION::Statistics::print_stats_short("tensor_gt_mul_test", run_time_stats, comm_stats);
      std::cout << "Execution time string:" << time_str << "\n";
      double exec_time = std::stod(time_str);
      std::cout << "Execution time:" << exec_time << "\n";
      file2 << "Execution time - " << exec_time << "msec\n";
      file2.close();

      std::ofstream file1;
      file1.open(t1, std::ios_base::app);
      file1 << exec_time;
      file1 << "\n";
      file1.close();
    }
   }
   catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}