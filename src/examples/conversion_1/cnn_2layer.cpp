// ./bin/cnn_2layer 
//3683 - 15000 iterations - 73.6     63% - 1000 iterations  
//Relu - 1000 iterations (73.58) 15000 - 74.78
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <utility/new_fixed_point.h>
#include "utility/linear_algebra.h"

#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <random>

namespace po = boost::program_options;
using namespace std;
int frac_bits = 13;

uint64_t indicator(uint64_t input) {
    uint64_t msb = (uint64_t)1 << 63;
    if (input & msb) {
        return 0;
    } else {
       return 8192;
    }
}

void swap_(uint64_t &a, uint64_t &b) {
    uint64_t temp = a;
    a = b;
    b = temp;
}

uint64_t actual_sigmoid(uint64_t num, int frac_bits) {
    float decoded_num = MOTION::new_fixed_point::decode<std::uint64_t, float>(num, frac_bits);
    float sigmoid = 1 / (1 + exp(-1 * decoded_num));
    return MOTION::new_fixed_point::encode<std::uint64_t, float>(sigmoid, frac_bits);
}

uint64_t relu(uint64_t input, int frac_bits) {
    uint64_t msb = (uint64_t)1 << 63;
    if (input & msb) {
        return 0;
    } else {
       return input;
    }
}

void print_matrix(vector<uint64_t>input, string text, bool print = false, int start = 0, int end = 0) {
    if(print) {
        cout << text << ": " << endl;
        if(end == 0) end = input.size();
        for(int i = start; i < end; i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(input[i], frac_bits) << " ";
        }
        cout << endl;
    }
}

vector<vector<uint64_t>> dilate_matrix(vector<vector<uint64_t>>kernel_segments, int channels, int kernels, vector<int> strides, int rows, int cols) {
    vector<vector<uint64_t>>dilated_kernels;
    for(auto kernel: kernel_segments) {
        vector<uint64_t>dilated_kernel;        
        int dilated_columns = cols + (cols - 1) * (strides[1] - 1);
        for(int channel = 0; channel < channels; channel++) {
            int r = 0;
            while(r < rows) {
                int c = 0;
                while(c < cols) {
                    dilated_kernel.push_back(kernel[(channel * rows * cols) + (r * cols + c)]);
                    if(c != cols - 1) {
                        for(int i = 0; i < strides[1] - 1; i++) {
                            dilated_kernel.push_back(0);
                        }
                    }
                    c++;
                }
                if(r != rows - 1) {
                    for(int s = 0; s < strides[0] - 1; s++) {
                        for(int i = 0; i < dilated_columns; i++) {
                            dilated_kernel.push_back(0);
                        }
                    }
                }
                r++;
            }
        }
        dilated_kernels.push_back(dilated_kernel);
    }
    return dilated_kernels;
}

vector<uint64_t>convolution(vector<uint64_t>input, vector<uint64_t>weights, vector<uint64_t>bias, int kernels, int channels,
    int rows, int cols, vector<int> pads, vector<int> strides, int img_rows, int img_cols, bool dilate = false) {
    vector<vector<uint64_t>> kernel_segments;
     
    int temp = 0;
    for (int i=0; i<kernels; i++) {
        auto first = weights.begin() + temp;
        temp+=channels*rows*cols;
        auto last = weights.begin() + temp;
        vector<uint64_t> kernel(first, last);
        kernel_segments.push_back(kernel);
    }

    if(dilate) {
        kernel_segments = dilate_matrix(kernel_segments, channels, kernels, strides, rows, cols);
        rows = rows + (rows - 1) * (strides[0] - 1);
        cols = cols + (cols - 1) * (strides[1] - 1);
        strides = {1, 1};
    }

    auto encoded_zero = MOTION::new_fixed_point::encode<uint64_t, float>(0, frac_bits);

    vector<uint64_t> image;
    for (int c=0; c<channels; c++) {
        for (int i=0; i<pads[0]; i++) {
            for (int j=0; j<img_cols+pads[1]+pads[3]; j++)
                image.push_back(encoded_zero);
        }

        for (int i=0; i<img_rows; i++) {
            for (int j=0; j<pads[1]; j++)
                image.push_back(encoded_zero);
            for (int j=0; j<img_cols; j++)
                image.push_back(input[c*img_rows*img_cols + i*img_cols+ j]);
            for (int j=0; j<pads[3]; j++)
                image.push_back(encoded_zero);
        }

        for (int i=0; i<pads[2]; i++) {
            for (int j=0; j<img_cols+pads[1]+pads[3]; j++)
                image.push_back(encoded_zero);
        }
    }
    img_rows += pads[0] + pads[2];
    img_cols += pads[1] + pads[3];

    uint64_t output_chnls = kernels;
    uint64_t output_rows = (img_rows - rows + strides[0]) / strides[0];
    uint64_t output_columns = (img_cols - cols + strides[1]) / strides[1];

    vector<vector<uint64_t>> image_segments(output_rows*output_columns, vector<uint64_t>(channels*rows*cols));
    
    for (unsigned int i=0; i<output_rows; i++) {
        for (unsigned int j=0; j<output_columns; j++) {
            unsigned int row_start = i*strides[0];
            unsigned int col_start = j*strides[1];
            for (unsigned int k=0; k<channels; k++) {
                for (unsigned int l=0; l<rows*cols; l++) {
                    unsigned int row = row_start + l/cols;
                    unsigned int col = col_start + l%cols;
                    image_segments[i*output_columns + j][k*rows*cols + l] = image[k*img_rows*img_cols + row*img_cols + col];
                }
            }
        }
    }
    vector<uint64_t> output(kernels*output_rows*output_columns);
    int j=0;
    for (int k=0; k<kernels; k++) {
        for (int i=0; i<output_rows*output_columns; i++) {
            output[j] = (MOTION::matrix_multiply(1, kernel_segments[k].size(), 1, kernel_segments[k], image_segments[i]))[0];
            output[j] = MOTION::new_fixed_point::decode<uint64_t, float>(output[j], frac_bits);
            output[j] += bias[k];
            j++;
        }
    }
    return output;
}

vector<uint64_t>weights_multi_channel_convolution(vector<uint64_t>image, vector<uint64_t>weights, int image_channels, int weight_channels,
    int rows, int cols, vector<int>pads, vector<int>strides, int img_rows, int img_cols, bool dilate = false) {
    
    vector<vector<uint64_t>>weight_channel_segments;
    int temp = 0;
    for (int i = 0; i < weight_channels; i++) {
        auto first = weights.begin() + temp;
        temp += rows * cols;
        auto last = weights.begin() + temp;
        vector<uint64_t>channel(first, last);
        weight_channel_segments.push_back(channel);
    }
    
    vector<vector<uint64_t>>image_channel_segments;
    temp = 0;
    for (int i = 0; i < image_channels; i++) {
        auto first = image.begin() + temp;
        temp += img_rows * img_cols;
        auto last = image.begin() + temp;
        vector<uint64_t>channel(first, last);
        image_channel_segments.push_back(channel);
    }
    
    vector<vector<uint64_t>>unflattened_output;
    for(auto weight_channel: weight_channel_segments) {
        for(auto image_channel: image_channel_segments) {
            unflattened_output.push_back(convolution(image_channel, weight_channel, {0}, 1, 1, rows, cols, pads, strides, img_rows, img_cols, dilate));
        }
    }

    vector<uint64_t>output;
    for(auto row: unflattened_output) {
        for(auto num: row) {
            output.push_back(num);
        }
    }
    return output;
}

vector<uint64_t>input_multi_channel_convolution(vector<uint64_t>image, vector<uint64_t>weights, int image_channels, int weight_channels,
    int rows, int cols, vector<int>pads, vector<int>strides, int img_rows, int img_cols, bool dilate = false) {
    
    vector<vector<uint64_t>>weight_channel_segments;
    int temp = 0;
    for(int c = 0; c < weight_channels * image_channels; c++) {
        auto first = weights.begin() + temp;
        temp += rows * cols;
        auto last = weights.begin() + temp;
        vector<uint64_t>channel(first, last);
        weight_channel_segments.push_back(channel);
    }

    //rotate the weight channels by 180
    for(int channel = 0; channel < weight_channel_segments.size(); channel++) {
        for(int r = 0; r < rows / 2; r++) {
            for(int c = 0; c < cols; c++) {
                swap_(weight_channel_segments[channel][r * cols + c], weight_channel_segments[channel][(rows - 1 - r) * cols + c]);
            }
        }
    }

    for(int channel = 0; channel < weight_channel_segments.size(); channel++) {
        for(int c = 0; c < cols / 2; c++) {
            for(int r = 0; r < rows; r++) {
                swap_(weight_channel_segments[channel][r * cols + c], weight_channel_segments[channel][r * cols + (cols - 1 - c)]);
            }
        }
    }

    vector<int>pad = {rows - 1, cols - 1, rows - 1, cols - 1};
    vector<uint64_t>padded_image;
    for (int c = 0; c < image_channels; c++) {
        for (int i = 0; i < pad[0]; i++) {
            for (int j = 0; j < img_cols + pad[1] + pad[3]; j++)
                padded_image.push_back(0);
        }

        for (int i = 0; i < img_rows; i++) {
            for (int j = 0; j < pad[1]; j++)
                padded_image.push_back(0);
            for (int j = 0; j < img_cols; j++)
                padded_image.push_back(image[c * img_rows * img_cols + i * img_cols + j]);
            for (int j = 0; j < pad[3]; j++)
                padded_image.push_back(0);
        }

        for (int i = 0; i < pad[2]; i++) {
            for (int j = 0; j < img_cols + pad[1] + pad[3]; j++)
                padded_image.push_back(0);
        }
    }

    img_rows += pad[0] + pad[2];
    img_cols += pad[1] + pad[3];

    vector<vector<uint64_t>>image_channel_segments;
    temp = 0;
    for (int i = 0; i < image_channels; i++) {
        auto first = padded_image.begin() + temp;
        temp += img_rows * img_cols;
        auto last = padded_image.begin() + temp;
        vector<uint64_t>channel(first, last);
        image_channel_segments.push_back(channel);
    }

    vector<vector<uint64_t>>convolution_output;
    for(int channel = 0; channel < weight_channels * image_channels; channel++) {
        convolution_output.push_back(convolution(image_channel_segments[channel / weight_channels], weight_channel_segments[channel], {0}, 1, 1, rows, cols, pads, strides, img_rows, img_cols, dilate));
    }

    uint64_t output_rows = (img_rows - rows + strides[0]) / strides[0];
    uint64_t output_cols = (img_cols - cols + strides[1]) / strides[1];
    
    vector<vector<uint64_t>>added_output(weight_channels, vector<uint64_t>(output_rows * output_cols));
    for(int channel = 0; channel < weight_channels * image_channels; channel++) {
        for(int i = 0; i < output_rows * output_cols; i++) {
            added_output[channel % weight_channels][i] += convolution_output[channel][i]; 
        } 
    }

    vector<uint64_t>output;
    for(auto row: added_output) {
        for(auto num: row) {
            output.push_back(num);
        }
    }
    return output;
}


uint64_t sigmoid(uint64_t dot_product, int frac_bits) {
    auto encoded_threshold = MOTION::new_fixed_point::encode<std::uint64_t, float>(0.5, frac_bits);
    auto neg_encoded_threshold = MOTION::new_fixed_point::encode<std::uint64_t, float>(-0.5, frac_bits);

    uint64_t msb = (uint64_t)1 << 63;

    if (dot_product & msb) {
        if (dot_product < neg_encoded_threshold || dot_product == neg_encoded_threshold) {
            return 0;
        } else {
            return dot_product + encoded_threshold;
        }
    } else {
        if (dot_product > encoded_threshold || dot_product == encoded_threshold) {
            return MOTION::new_fixed_point::encode<std::uint64_t, float>(1, frac_bits);
        } else {
            return dot_product + encoded_threshold;
        }
    }
}

void add_bias(vector<uint64_t>&cnn_output, vector<uint64_t>bias) {
    int channel_size = cnn_output.size() / bias.size();
    for(int i = 0; i < cnn_output.size(); i++) {
        cnn_output[i] += bias[(int)i / channel_size];
    }
}

vector<uint64_t> bias_gradients(vector<uint64_t>dLbydZ, int num_channels) {
    int channel_size = dLbydZ.size() / num_channels;
    vector<uint64_t>dLbydB(num_channels, 0);
    for(int i = 0; i < dLbydZ.size(); i++) {
        dLbydB[(int)i / channel_size] += dLbydZ[i]; 
    }
    return dLbydB;
}


class CNN {
    public:
    vector<uint64_t>image;
    vector<uint64_t>weights1;
    vector<uint64_t>weights2;
    vector<uint64_t>weights3;
    vector<uint64_t>bias1;
    vector<uint64_t>bias2;
    vector<uint64_t>bias3;
    vector<uint64_t>dLbydW1;
    vector<uint64_t>dLbydW2;
    vector<uint64_t>dLbydW3;
    vector<vector<uint64_t>>dLbydW1_batches;
    vector<vector<uint64_t>>dLbydW2_batches;
    vector<vector<uint64_t>>dLbydW3_batches;
    vector<uint64_t>dLbydB1;
    vector<uint64_t>dLbydB2;
    vector<uint64_t>dLbydB3;
    vector<vector<uint64_t>>dLbydB1_batches;
    vector<vector<uint64_t>>dLbydB2_batches;
    vector<vector<uint64_t>>dLbydB3_batches;
    vector<uint64_t>target;
    vector<uint64_t>cnn_layer1_output;
    vector<uint64_t>sigmoid_layer1_output;
    vector<uint64_t>cnn_layer2_output;
    vector<uint64_t>sigmoid_layer2_output;
    vector<uint64_t>fully_connected_layer1_output;
    vector<uint64_t>sigmoid_layer3_output;
    vector<uint64_t>output;

    int target_label;
    int batch_size;

    void set_batch_size(int size) {
        batch_size = size;
    }

    void initialize_random_weights(vector<int>rows, vector<int>cols, vector<int>channels, vector<int>kernels) {
        weights1.assign(rows[0] * cols[0] * channels[0] * kernels[0], 0);
        weights2.assign(rows[1] * cols[1] * channels[1] * kernels[1], 0);
        weights3.assign(rows[2] * cols[2] * channels[2] * kernels[2], 0);
        for(int i = 0; i < 3; i++) {
            random_device rd; 
            mt19937 gen(rd()); 
            uint64_t minNumber = 0;
            uint64_t maxNumber = MOTION::new_fixed_point::encode<uint64_t, float>(0.25, 13);
            uniform_int_distribution<uint64_t> distribution(minNumber, maxNumber);
            for(int j = 0; j < rows[i] * cols[i] * channels[i] * kernels[i]; j++) {   
                int x = distribution(gen);
                if(i == 0) {
                    weights1[j] = x;
                } else if(i == 1) {
                    weights2[j] = x;
                } else if(i == 2) {
                    weights3[j] = x;
                }
            }
        }
    }

    void initialize_random_biases(vector<int>rows, vector<int>cols, vector<int>channels, vector<int>kernels) {
        bias1.assign(rows[0] * cols[0] * channels[0] * kernels[0], 0);
        bias2.assign(rows[1] * cols[1] * channels[1] * kernels[1], 0);
        bias3.assign(rows[2] * cols[2] * channels[2] * kernels[2], 0);
        for(int i = 0; i < 3; i++) {
            random_device rd; 
            mt19937 gen(rd()); 
            uint64_t minNumber = 0;
            uint64_t maxNumber = MOTION::new_fixed_point::encode<uint64_t, float>(1, 13);
            uniform_int_distribution<uint64_t> distribution(minNumber, maxNumber);
            for(int j = 0; j < rows[i] * cols[i] * channels[i] * kernels[i]; j++) {   
                int x = distribution(gen);
                if(i == 0) {
                    bias1[j] = x;
                } else if(i == 1) {
                    bias2[j] = x;
                } else if(i == 2) {
                    bias3[j] = x;
                }
            }
        }
    }

    void get_input(int i) {
        vector<float>input_image;
        string home_dir = getenv("BASE_DIR");
        ifstream file;
        string path = home_dir + "/data/ImageProvider/sample_data/";
        
        random_device rd; 
        mt19937 gen(rd()); 
        uint64_t minNumber = 1;
        uint64_t maxNumber = 5000;
        uniform_int_distribution<uint64_t> distribution(minNumber, maxNumber);
        int x = distribution(gen);

        random_device rd_; 
        mt19937 gen_(rd_()); 
        uint64_t minNumber_ = 0;
        uint64_t maxNumber_ = 9;
        uniform_int_distribution<uint64_t> distribution_(minNumber_, maxNumber_);
        int y = distribution_(gen_);
        target_label = y;
        string input_path;
        input_path = path + "images_folder" + to_string(y) + "/X" + to_string(x) + ".csv";
        
        file.open(input_path);
        string str;
        while (std::getline(file, str)) {
            stringstream obj(str);
            string temp;
            while (std::getline(obj, temp, ',')) {
                auto input = std::stof(temp);
                input_image.push_back(input);
            }
        }
        file.close();
        
        vector<uint64_t>encoded_input_image(input_image.size(), 0);
        transform(input_image.begin(), input_image.end(), encoded_input_image.begin(), [](auto j) {
            return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
        });
        image = encoded_input_image;
    }

    void get_target(int target_label_) {
        vector<float>target_(10, 0);
        target_[target_label % 10] = 1;
        vector<uint64_t> encoded_target(target_.size(), 0);
        transform(target_.begin(), target_.end(), encoded_target.begin(), [](auto j) {
            return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
        });
        target = encoded_target;
    }

    void forward_pass() {
        print_matrix(image, "Image");
        print_matrix(weights1, "Weights 1", true, 0, 50);
        cnn_layer1_output = convolution(image, weights1, bias1, 5, 1, 5, 5, {1, 1, 0, 0}, {2, 2}, 28, 28, false);
        print_matrix(cnn_layer1_output, "CNN layer 1 output");
        print_matrix(bias1, "Bias 1");
        //add_bias(cnn_layer1_output, bias1);
        print_matrix(cnn_layer1_output, "CNN layer 1 output after bias addition");
        sigmoid_layer1_output.assign(cnn_layer1_output.size(), 0);
        transform(cnn_layer1_output.begin(), cnn_layer1_output.end(), sigmoid_layer1_output.begin(),
                  [frac_bits] (auto j)  { return relu(j, frac_bits); });
        print_matrix(sigmoid_layer1_output, "Sigmoid layer 1 output");
        print_matrix(weights2, "Weights2", false, 0, 50);
        cnn_layer2_output = convolution(sigmoid_layer1_output, weights2, bias2, 4, 5, 3, 3, {0, 0, 0, 0}, {1, 1}, 13, 13, false);        
        print_matrix(cnn_layer2_output, "CNN layer 2 output");
        print_matrix(bias2, "Bias 2");
        //add_bias(cnn_layer2_output, bias2);
        print_matrix(cnn_layer2_output, "CNN layer 2 output after bias addition");
        sigmoid_layer2_output.assign(cnn_layer2_output.size(), 0);
        transform(cnn_layer2_output.begin(), cnn_layer2_output.end(), sigmoid_layer2_output.begin(),
                  [frac_bits] (auto j)  { return relu(j, frac_bits); });
        print_matrix(sigmoid_layer2_output, "Sigmoid layer 2 output");
        fully_connected_layer1_output = MOTION::matrix_multiply(1, sigmoid_layer2_output.size(), 10, sigmoid_layer2_output, weights3);
        print_matrix(weights3, "Weights3", false, 0, 50);
        transform(fully_connected_layer1_output.begin(), fully_connected_layer1_output.end(), fully_connected_layer1_output.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 }); 
        print_matrix(fully_connected_layer1_output, "Fully connected layer 1 output");
        print_matrix(bias3, "Bias 3");
        for(int i = 0; i < fully_connected_layer1_output.size(); i++) {
            fully_connected_layer1_output[i] += bias3[i];
        }
        print_matrix(fully_connected_layer1_output, "Fully connected layer 1 output after bias addition");
        sigmoid_layer3_output.assign(fully_connected_layer1_output.size(), 0);
        transform(fully_connected_layer1_output.begin(), fully_connected_layer1_output.end(), sigmoid_layer3_output.begin(),
                  [frac_bits] (auto j)  { return sigmoid(j, frac_bits); });       
        print_matrix(sigmoid_layer3_output, "Sigmoid layer 3 output");
        output = sigmoid_layer3_output;
    }

    void backward_pass() {
        print_matrix(output, "Output");
        print_matrix(target, "Target");
        vector<uint64_t>diff;
        for(int i = 0; i < target.size(); i++) {
            diff.push_back(output[i] - target[i]);
        }
        dLbydB3 = diff;
        print_matrix(dLbydB3, "dLbydB3");
        dLbydW3 = MOTION::matrix_multiply(sigmoid_layer2_output.size(), 1, 10, sigmoid_layer2_output, diff);
        transform(dLbydW3.begin(), dLbydW3.end(), dLbydW3.begin(),
            [frac_bits](auto j) {
                return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
            });
        print_matrix(dLbydW3, "dLbydW3", false, 0, 50);
        vector<uint64_t>weights3_transpose;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < sigmoid_layer2_output.size(); j++) {
                auto data = weights3[j * 10 + i];
                weights3_transpose.push_back(data);
            }
        }
        vector<uint64_t>dLbydF = MOTION::matrix_multiply(1, 10, sigmoid_layer2_output.size(), diff, weights3_transpose);
        transform(dLbydF.begin(), dLbydF.end(), dLbydF.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        vector<uint64_t>dA2bydZ2(dLbydF.size(), 0);
        uint64_t encoded_one = MOTION::new_fixed_point::encode<std::uint64_t, float>(1, frac_bits);
        for(int i = 0; i < dLbydF.size(); i++) {
            dA2bydZ2[i] = indicator(cnn_layer2_output[i]);//MOTION::new_fixed_point::decode<std::uint64_t, float>(sigmoid_layer2_output[i] * (encoded_one - sigmoid_layer2_output[i]), frac_bits);
        }
        print_matrix(dA2bydZ2, "dA2bydZ2");
        vector<uint64_t>dLbydZ2(dLbydF.size());
        transform(std::begin(dLbydF), std::end(dLbydF),
                    std::begin(dA2bydZ2), std::begin(dLbydZ2),
                    std::multiplies{});

        transform(dLbydZ2.begin(), dLbydZ2.end(), dLbydZ2.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        dLbydB2 = bias_gradients(dLbydZ2, bias2.size());
        print_matrix(dLbydB2, "dLbydB2");
        dLbydW2 = weights_multi_channel_convolution(sigmoid_layer1_output, dLbydZ2, 5, 4, 3, 3, {0, 0, 0, 0}, {1, 1}, 13, 13, false);
        print_matrix(dLbydW2, "dLbydW2", false, 0, 50);
        vector<uint64_t>dLbydA1 = input_multi_channel_convolution(dLbydZ2, weights2, 4, 5, 3, 3, {0, 0, 0, 0}, {1, 1}, 11, 11, false);
        print_matrix(dLbydA1, "dLbydA1");
        vector<uint64_t>dLbydZ1(dLbydA1.size());
        vector<uint64_t>dA1bydZ1(sigmoid_layer1_output.size(), 0);
        for(int i = 0; i < sigmoid_layer1_output.size(); i++) {
            dA1bydZ1[i] = indicator(cnn_layer1_output[i]);//MOTION::new_fixed_point::decode<std::uint64_t, float>(sigmoid_layer1_output[i] * (encoded_one - sigmoid_layer1_output[i]), frac_bits);
        }
        print_matrix(dA1bydZ1, "dA1bydZ1");
        transform(std::begin(dLbydA1), std::end(dLbydA1),
                    std::begin(dA1bydZ1), std::begin(dLbydZ1),
                    std::multiplies{});

        transform(dLbydZ1.begin(), dLbydZ1.end(), dLbydZ1.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        print_matrix(dLbydZ1, "dLbydZ1");
        dLbydB1 = bias_gradients(dLbydZ1, bias1.size());
        print_matrix(dLbydB1, "dLbydB1");

        dLbydW1 = convolution(image, dLbydZ1, {0, 0, 0, 0, 0}, 5, 1, 13, 13, {1, 1, 0, 0}, {2, 2}, 28, 28, true); 
        print_matrix(dLbydW1, "dLbydW1", true, 0, 50);
    }

    void update_weights() {
        dLbydW1_batches.push_back(dLbydW1);           
        dLbydW2_batches.push_back(dLbydW2);
        dLbydW3_batches.push_back(dLbydW3);
    }

    void update_biases() {
        dLbydB1_batches.push_back(dLbydB1);           
        dLbydB2_batches.push_back(dLbydB2);
        dLbydB3_batches.push_back(dLbydB3);
    }

    void update_batch_weights() {
        vector<uint64_t>dLbydW1_avg(dLbydW1_batches[0].size(), (uint64_t)0);
        vector<uint64_t>dLbydW2_avg(dLbydW2_batches[0].size(), (uint64_t)0);  
        vector<uint64_t>dLbydW3_avg(dLbydW3_batches[0].size(), (uint64_t)0);  

        for(int c = 0; c < dLbydW1_batches[0].size(); c++) {
            for(int r = 0; r < dLbydW1_batches.size(); r++) {
                dLbydW1_avg[c] += dLbydW1_batches[r][c];
            }
        }
        
        for(int c = 0; c < dLbydW2_batches[0].size(); c++) {
            for(int r = 0; r < dLbydW2_batches.size(); r++) {
                dLbydW2_avg[c] += dLbydW2_batches[r][c];
            }
        }

        for(int c = 0; c < dLbydW3_batches[0].size(); c++) {
            for(int r = 0; r < dLbydW3_batches.size(); r++) {
                dLbydW3_avg[c] += dLbydW3_batches[r][c];
            }
        }

        uint64_t rate = MOTION::new_fixed_point::encode<std::uint64_t, float>(0.05 / batch_size, 13);
        transform(dLbydW3_avg.begin(), dLbydW3_avg.end(),
                        dLbydW3_avg.begin(), [rate](auto j) { return rate * j; });
        transform(dLbydW3_avg.begin(), dLbydW3_avg.end(),
                        dLbydW3_avg.begin(), [frac_bits](auto j) {
                        return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                        });
        vector<std::uint64_t> new_weights3(weights3.size(), (uint64_t)0);
        transform(weights3.begin(), weights3.end(), dLbydW3_avg.begin(),
                        new_weights3.begin(), std::minus{});
        for(int i = 0; i < new_weights3.size(); i++)
        {
            weights3[i] = new_weights3[i];
        }

        transform(dLbydW2_avg.begin(), dLbydW2_avg.end(),
                        dLbydW2_avg.begin(), [rate](auto j) { return rate * j; });
        transform(dLbydW2_avg.begin(), dLbydW2_avg.end(),
                        dLbydW2_avg.begin(), [frac_bits](auto j) {
                        return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                        });
        vector<std::uint64_t> new_weights2(weights2.size(), (uint64_t)0);
        transform(weights2.begin(), weights2.end(), dLbydW2_avg.begin(),
                        new_weights2.begin(), std::minus{});
        for(int i=0;i<new_weights2.size();i++)
        {
            weights2[i]=new_weights2[i];
        }

        transform(dLbydW1_avg.begin(), dLbydW1_avg.end(),
                        dLbydW1_avg.begin(), [rate](auto j) { return rate * j; });
        transform(dLbydW1_avg.begin(), dLbydW1_avg.end(),
                        dLbydW1_avg.begin(), [frac_bits](auto j) {
                        return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                        });
        vector<std::uint64_t> new_weights1(weights1.size(), (uint64_t)0);
        transform(weights1.begin(), weights1.end(), dLbydW1_avg.begin(),
                        new_weights1.begin(), std::minus{});
        for(int i = 0; i < new_weights1.size(); i++)
        {
            weights1[i] = new_weights1[i];
        }
        dLbydW1_batches.clear();           
        dLbydW2_batches.clear();
        dLbydW3_batches.clear();
    }

    void update_batch_biases() {
        vector<uint64_t>dLbydB1_avg(dLbydB1_batches[0].size(), (uint64_t)0);
        vector<uint64_t>dLbydB2_avg(dLbydB2_batches[0].size(), (uint64_t)0);  
        vector<uint64_t>dLbydB3_avg(dLbydB3_batches[0].size(), (uint64_t)0);  
        print_matrix(dLbydB1_avg, "dLbydB1_avg");
        for(int c = 0; c < dLbydB1_batches[0].size(); c++) {
            for(int r = 0; r < dLbydB1_batches.size(); r++) {
                dLbydB1_avg[c] += dLbydB1_batches[r][c];
            }
        }
        
        for(int c = 0; c < dLbydB2_batches[0].size(); c++) {
            for(int r = 0; r < dLbydB2_batches.size(); r++) {
                dLbydB2_avg[c] += dLbydB2_batches[r][c];
            }
        }

        for(int c = 0; c < dLbydB3_batches[0].size(); c++) {
            for(int r = 0; r < dLbydB3_batches.size(); r++) {
                dLbydB3_avg[c] += dLbydB3_batches[r][c];
            }
        }

        print_matrix(dLbydB1_batches[0], "dLbydB1[0]");
        print_matrix(dLbydB1_avg, "dLbydB1_avg");

        uint64_t rate = MOTION::new_fixed_point::encode<std::uint64_t, float>(0.0005 / batch_size, 13);
        transform(dLbydB3_avg.begin(), dLbydB3_avg.end(),
                        dLbydB3_avg.begin(), [rate](auto j) { return rate * j; 
                        });
        transform(dLbydB3_avg.begin(), dLbydB3_avg.end(),
                        dLbydB3_avg.begin(), [frac_bits](auto j) {
                        return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                        });

        print_matrix(dLbydB1_avg, "dLbydB1_avg * rate");

        vector<std::uint64_t> new_bias3(bias3.size(), (uint64_t)0);
        transform(bias3.begin(), bias3.end(), dLbydB3_avg.begin(),
                        new_bias3.begin(), std::minus{});
        for(int i = 0; i < new_bias3.size(); i++)
        {
            bias3[i] = new_bias3[i];
        }

        transform(dLbydB2_avg.begin(), dLbydB2_avg.end(),
                        dLbydB2_avg.begin(), [rate](auto j) { return rate * j; });
        transform(dLbydB2_avg.begin(), dLbydB2_avg.end(),
                        dLbydB2_avg.begin(), [frac_bits](auto j) {
                        return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                        });
        vector<std::uint64_t> new_bias2(bias2.size(), (uint64_t)0);
        transform(bias2.begin(), bias2.end(), dLbydB2_avg.begin(),
                        new_bias2.begin(), std::minus{});
        for(int i=0;i<new_bias2.size();i++)
        {
            bias2[i]=new_bias2[i];
        }

        transform(dLbydB1_avg.begin(), dLbydB1_avg.end(),
                        dLbydB1_avg.begin(), [rate](auto j) { return rate * j; });
        transform(dLbydB1_avg.begin(), dLbydB1_avg.end(),
                        dLbydB1_avg.begin(), [frac_bits](auto j) {
                        return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                        });
        vector<std::uint64_t> new_bias1(bias1.size(), (uint64_t)0);
        transform(bias1.begin(), bias1.end(), dLbydB1_avg.begin(),
                        new_bias1.begin(), std::minus{});
        for(int i = 0; i < new_bias1.size(); i++)
        {
            bias1[i] = new_bias1[i];
        }
        dLbydB1_batches.clear();           
        dLbydB2_batches.clear();
        dLbydB3_batches.clear();
    }

    int get_test_input(int i) {
        string home_dir = getenv("BASE_DIR");
        ifstream file;
        string path = home_dir + "/data/ImageProvider/labelled_test_data";
        string input_path;
        input_path = path + "/X" + std::to_string(i) + ".csv";
        vector<float>unencoded_input;
        file.open(input_path);
        string str;
        int test_label;
        int line = 0;
        while (getline(file, str)) {
        std::stringstream obj(str);
        std::string temp;
        while (getline(obj, temp, ',')) {
            if(line == 0) {
            auto input = stof(temp);
            test_label = input;
            } else {
            auto input = stof(temp);
            unencoded_input.push_back(input);
            }
        }
        line++;
        }
        file.close();
  
        vector<uint64_t> encoded_input_image(unencoded_input.size(), 0);
        transform(unencoded_input.begin(), unencoded_input.end(), encoded_input_image.begin(), [](auto j) {
            return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
        });
        image = encoded_input_image;
        return test_label;
    }

    int get_prediction() {
        vector<float>decoded_output(output.size());
        print_matrix(output, "Output");
        transform(output.begin(), output.end(), decoded_output.begin(), [](auto j) {
            return MOTION::new_fixed_point::decode<uint64_t, float>(j, 13);
        });
        float max = INT_MIN;
        int argmax = 0;
        for(int i = 0; i < decoded_output.size(); i++) {
            if(max < decoded_output[i]) {
                max = decoded_output[i];
                argmax = i;
            }
        }
        return argmax;
    }

    void save_weights(string text) {
        string home_dir = getenv("BASE_DIR");
        ofstream out_file1;
        string path1 = home_dir+"/build_debwithrelinfo_gcc/cnn_2layer_w1";

        out_file1.open(path1, ios_base::app); 
        out_file1 << text << endl;   
        for (int i = 0; i < weights1.size(); i++) {
            out_file1 << MOTION::new_fixed_point::decode<uint64_t, float>(weights1[i], 13) <<",";
        }
        out_file1 << "\n";
        out_file1.close(); 

        ofstream out_file2;
        string path2 = home_dir+"/build_debwithrelinfo_gcc/cnn_2layer_w2";

        out_file2.open(path2, ios_base::app);  
        out_file2 << text << endl;   
        for (int i = 0; i < weights2.size(); i++) {
            out_file2 << MOTION::new_fixed_point::decode<uint64_t, float>(weights2[i], 13) <<",";
        }
        out_file2 << "\n";
        out_file2.close(); 

        ofstream out_file3;
        out_file3 << text << endl;   
        string path3 = home_dir+"/build_debwithrelinfo_gcc/cnn_2layer_w3";

        out_file3.open(path3, ios_base::app);    
        for (int i = 0; i < weights3.size(); i++) {
            out_file3 << MOTION::new_fixed_point::decode<uint64_t, float>(weights3[i], 13) <<",";
        }
        out_file3 << "\n";
        out_file3.close(); 
    }
    
};


int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 20;
    int batch_size = 1;
    CNN cnn = CNN();    
    cnn.initialize_random_weights({5, 3, 484}, {5, 3, 10}, {1, 5, 1}, {5, 4, 1}); //rows, columns, channels, kernels
    cnn.initialize_random_biases({1, 1, 1}, {5, 4, 10}, {1, 1, 1}, {1, 1, 1}); //rows, columns, channels, kernels
    cnn.set_batch_size(batch_size);
    for(int i = 0; i < iterations; i++) {
        if(i % 10 == 0) cout << "\nIteration: " << i << endl;
        for(int j = 0; j < batch_size; j++) {
            cnn.get_input(i);
            cnn.get_target(i);
            cnn.forward_pass();
            cnn.backward_pass();
            cnn.update_weights();
            cnn.update_biases();
        }
        cnn.update_batch_weights();
        cnn.update_batch_biases();
    }
    //cnn.save_weights("5 iterations with 20 batch size and relu, relu, sigmoid");
    cout << "Testing: " << endl;
    int accuracy = 0;
    for(int i = 1; i < 0; i++) {
        int test_label = cnn.get_test_input(i);
        cnn.forward_pass();
        int prediction = cnn.get_prediction();
        cout << "Test label:" << test_label << "  Prediction: " << prediction << endl;
        if(prediction == test_label) accuracy++;
    }
    cout << "Accuracy: " << accuracy << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end - start);
    cout << "Time taken: " << duration.count() << " minutes" << std::endl;

    /*
    vector<uint64_t>input = {1, 2, 3, 4, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1};
    vector<uint64_t>weight = {1, 1, 0, 0, 0, 0, 1, 1};
    transform(input.begin(), input.end(), input.begin(), [](auto j) {
        return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
    });
    transform(weight.begin(), weight.end(), weight.begin(), [](auto j) {
        return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
    });
    print_matrix(weights_multi_channel_convolution(input, weight, 2, 2, 2, 2, {0, 0, 0, 0}, {1, 1}, 4, 4), ""); 
    

    vector<uint64_t>input = {1, 1, 0, 1, 1, 0, 1, 1};
    vector<uint64_t>weight = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    transform(input.begin(), input.end(), input.begin(), [](auto j) {
        return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
    });
    transform(weight.begin(), weight.end(), weight.begin(), [](auto j) {
        return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
    });
    print_matrix(input_multi_channel_convolution(input, weight, 2, 2, 3, 3, {0, 0, 0, 0}, {1, 1}, 2, 2), "");
    

   vector<float>un_nums = {-1.5, -1, -0.75, -0.5, -0.3, 0, 0.3, 0.5, 0.75, 1, 1.5};
   vector<uint64_t>nums(un_nums.size(), 0);
   
   transform(un_nums.begin(), un_nums.end(), nums.begin(), [](auto j) {
        return MOTION::new_fixed_point::encode<uint64_t, float>(j, 13);
    });
    for(auto num: nums) {
        cout << MOTION::new_fixed_point::decode<uint64_t, float>(actual_sigmoid(num, 13), 13) << " ";
    }
    cout << endl;
    */
    return EXIT_SUCCESS;
}