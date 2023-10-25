// ./bin/cnn
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

class ConvolutionLayer {
    public:
    vector<uint64_t>input;
    vector<uint64_t>weights;
    vector<uint64_t>output;
    vector<uint64_t>dLbydW;

    vector<uint64_t> forward_pass(vector<uint64_t>cnn_input, vector<uint64_t>input_weights) {
        input = cnn_input;
        weights = input_weights;
        output = convolution(input, weights, {0, 0, 0, 0, 0}, 5, 1, 5, 5, {1, 1, 0, 0}, {2, 2}, 28, 28);
        return output;
    }

    vector<uint64_t> backward_pass(vector<uint64_t>dLbydF) {
        dLbydW = convolution(input, dLbydF, {0, 0, 0, 0, 0}, 5, 1, 13, 13, {1, 1, 0, 0}, {2, 2}, 28, 28, true);
        return dLbydW;
    }
};

class FlattenLayer {
    public:
    vector<uint64_t>input;
    vector<uint64_t>output;
    vector<uint64_t>dLbydF;

    vector<uint64_t> forward_pass(vector<uint64_t>flatten_layer_input) {
        input = flatten_layer_input;
        output = input;
        return output;
    }
    
    vector<uint64_t> backward_pass(vector<uint64_t>dLbydP) {
        dLbydF = dLbydP; //Reshaped
        return dLbydF;
    }
};

class SigmoidLayer {
    public:
    vector<uint64_t>input;
    vector<uint64_t>output;
    vector<uint64_t>dLbydF;
    
    vector<uint64_t> forward_pass(vector<uint64_t>sigmoid_input) {
        input = sigmoid_input;
        output.assign(input.size(), 0);
        transform(input.begin(), input.end(), output.begin(),
                  [this, frac_bits] (uint64_t j)  { return this->sigmoid(j, frac_bits); });
        return output;
    }

    vector<uint64_t> backward_pass(vector<uint64_t>dLbydX2) {
        vector<uint64_t>dX2bydF(input.size(), 0);
        for(int i = 0; i < input.size(); i++) {
            dX2bydF[i] = MOTION::new_fixed_point::decode<std::uint64_t, float>(output[i] * (8192 - output[i]), 13);
        }
        dLbydF.assign(dLbydX2.size(), 0);

        transform(std::begin(dLbydX2), std::end(dLbydX2),
                    std::begin(dX2bydF), std::begin(dLbydF),
                    std::multiplies{});

        transform(dLbydF.begin(), dLbydF.end(), dLbydF.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        return dLbydF;
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
};

class ReluLayer {
    public:
    vector<uint64_t>input;
    vector<uint64_t>output;
    vector<uint64_t>dLbydF;

    vector<uint64_t> forward_pass(vector<uint64_t>relu_input) {
        input = relu_input;
        output.assign(input.size(), 0);
        transform(input.begin(), input.end(), output.begin(),
                  [this, frac_bits] (uint64_t j)  { return this->relu(j); });
        return output;
    }

    vector<uint64_t> backward_pass(vector<uint64_t>dLbydX2) {
        vector<uint64_t>dX2bydF(input.size(), 0);
        for(int i = 0; i < input.size(); i++) {
            dX2bydF[i] = output[i] > 0? 1: 0;
        }
        dLbydF.assign(dLbydX2.size(), 0);

        transform(std::begin(dLbydX2), std::end(dLbydX2),
                    std::begin(dX2bydF), std::begin(dLbydF),
                    std::multiplies{});

        transform(dLbydF.begin(), dLbydF.end(), dLbydF.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        return dLbydF;
    }
    
    uint64_t relu(uint64_t input) {
        uint64_t msb = (uint64_t)1 << 63;
        if (input & msb) {
            return 0;
        } else {
            return input;
        }   
    }
};

class FullyConnectedLayer {
    public:
    vector<uint64_t>input;
    vector<uint64_t>output;
    vector<uint64_t>dLbydW;
    vector<uint64_t>dLbydX;
    vector<uint64_t>weights;
    vector<uint64_t> forward_pass(vector<uint64_t>fc_layer_input, vector<uint64_t>input_weights) {
        input = fc_layer_input;
        weights = input_weights;
        output = MOTION::matrix_multiply(1, 845, 10, input, weights);
        transform(output.begin(), output.end(), output.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        return output;
    }

    vector<uint64_t> backward_pass(vector<uint64_t>prediction, vector<uint64_t>target, bool w, bool x) {
        vector<uint64_t>diff;
        for(int i = 0; i < target.size(); i++) {
            diff.push_back(prediction[i] - target[i]);
        }
        
        vector<uint64_t>input_transpose;
        for (int i = 0; i < 845; i++) {
            for (int j = 0; j < 1; j++) {
                auto data = input[j * 845 + i];
            input_transpose.push_back(data);
            }
        }

        vector<uint64_t>weights_transpose;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 845; j++) {
                auto data = weights[j * 10 + i];
            weights_transpose.push_back(data);
            }
        }
    
        if(w) {
            dLbydW = MOTION::matrix_multiply(845, 1, 10, input_transpose, diff);
            transform(dLbydW.begin(), dLbydW.end(), dLbydW.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
            return dLbydW;
        }
        dLbydX = MOTION::matrix_multiply(1, 10, 845, diff, weights_transpose);
        transform(dLbydX.begin(), dLbydX.end(), dLbydX.begin(),
                 [frac_bits](auto j) {
                   return MOTION::new_fixed_point::decode<std::uint64_t, float>(j, frac_bits);
                 });
        return dLbydX;
    }
};

class CNN {
    public:
    ConvolutionLayer cnn_layer1 = ConvolutionLayer();
    FlattenLayer flatten_layer = FlattenLayer();
    //SigmoidLayer sigmoid_layer1 = SigmoidLayer();
    ReluLayer relu_layer1 = ReluLayer();
    FullyConnectedLayer fully_connected_layer1 = FullyConnectedLayer();
    SigmoidLayer sigmoid_layer2 = SigmoidLayer();
    vector<uint64_t>image;
    vector<uint64_t>weights1;
    vector<uint64_t>weights2;
    vector<uint64_t>output;
    vector<uint64_t>dLbydW1;
    vector<uint64_t>dLbydW2;
    vector<vector<uint64_t>>dLbydW1_batches;
    vector<vector<uint64_t>>dLbydW2_batches;
    vector<uint64_t>target;
    int target_label;
    int batch_size;

    void set_batch_size(int size) {
        batch_size = size;
    }

    void initialize_random_weights(vector<int>rows, vector<int>cols, vector<int>channels, vector<int>kernels) {
        weights1.assign(rows[0] * cols[0] * channels[0] * kernels[0], 0);
        weights2.assign(rows[1] * cols[1] * channels[1] * kernels[1], 0);
        for(int i = 0; i < 2; i++) {
            random_device rd; 
            mt19937 gen(rd()); 
            uint64_t minNumber = 0;
            uint64_t maxNumber = 8192;
            uniform_int_distribution<uint64_t> distribution(minNumber, maxNumber);
            for(int j = 0; j < rows[i] * cols[i] * channels[i] * kernels[i]; j++) {   
                int x = distribution(gen);
                if(i == 0) {
                    weights1[j] = x;
                } else if(i == 1) {
                    weights2[j] = x;
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
        
        //cout << "\nInput path: " << input_path << "\n";
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
        vector<uint64_t>cnn_layer1_output = cnn_layer1.forward_pass(image, weights1);
        vector<uint64_t>flatten_layer_output = flatten_layer.forward_pass(cnn_layer1_output);        
        vector<uint64_t>relu_layer1_output = relu_layer1.forward_pass(flatten_layer_output);
        vector<uint64_t>fully_connected_layer1_output = fully_connected_layer1.forward_pass(relu_layer1_output, weights2); 
        vector<uint64_t>sigmoid_layer2_output = sigmoid_layer2.forward_pass(fully_connected_layer1_output);
       
        output = sigmoid_layer2_output;
        
        /*
        cout<<"Forward pass: "<<endl;
        cout<<"Image: ";
        for(int i = 0; i < image.size(); i++) {
            if(i % 28 == 0) cout << "\n";    
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(image[i], 13) << " ";
        }
        cout <<"\nWeights: ";
         for(int i = 0; i < weights1.size(); i++) {
            if(i % 5 == 0) cout << "\n";
            if(i % 25 == 0) cout << "\n";
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(weights1[i], 13) << " ";
        }
        cout <<"\nConvolution output: ";
        for(int i = 0; i < cnn_layer1_output.size(); i++) {
            if(i % 13 == 0) cout << "\n";
            if(i % 169 == 0) cout << "\n";
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(cnn_layer1_output[i], 13) << " ";
        }
        cout <<"\nFlatten output: \n";
        for(int i = 0; i < flatten_layer_output.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(flatten_layer_output[i], 13) << " ";
        }
        cout <<"\nSigmoid layer output: \n";
        for(int i = 0; i < sigmoid_layer1_output.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(sigmoid_layer1_output[i], 13) << " ";
        }
        cout <<"\n Weights2: ";
        for(int i = 0; i < weights2.size(); i++) {
            if(i % 10 == 0) cout << "\n";
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(weights2[i], 13) << " ";
        }
        cout<<"\nFully connected layer output decoded: \n";
        for(int i = 0; i < fully_connected_layer1_output.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(fully_connected_layer1_output[i], 13) << " ";
        }
        cout<<"\nSigmoid layer2 output: \n";
        for(int i = 0; i < sigmoid_layer2_output.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(sigmoid_layer2_output[i], 13) << " ";
        }
        cout << "\nTarget: \n";
        for(int i = 0; i < target.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(target[i], 13) << " ";
        }*/
    }

    void backward_pass() {
        /*cout << "\nTarget: \n";
        for(int i = 0; i < target.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(target[i], 13) << " ";
        }
        cout << "\nFoward pass output: \n";
        for(int i = 0; i < output.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(output[i], 13) << " ";
        }*/
        dLbydW2 = fully_connected_layer1.backward_pass(output, target, true, false);
        vector<uint64_t>dLbydX2 = fully_connected_layer1.backward_pass(output, target, false, true);
        vector<uint64_t>dLbydF = relu_layer1.backward_pass(dLbydX2);
        vector<uint64_t>dLbydZ = flatten_layer.backward_pass(dLbydF);
        dLbydW1 = cnn_layer1.backward_pass(dLbydZ);
        /*
        cout<<"\ndLbydW2 decoded: \n";
        for(int i = 100; i < 105; i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(dLbydW2[i], 13) << " ";
        }
        cout<<"\ndLbydX2 decoded: \n";
        for(int i = 0; i < dLbydX2.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(dLbydX2[i], 13) << " ";
        }
        cout << "\ndLbydF: \n";
        for(int i = 0; i < dLbydF.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(dLbydF[i], 13) << " ";
        }
        cout << "\ndLbydZ: \n";
        for(int i = 0; i < dLbydZ.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(dLbydZ[i], 13) << " ";
        }
        cout << "\ndLbydW1: \n";
        for(int i = 0; i < dLbydW1.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(dLbydW1[i], 13) << " ";
        }  */ 
    }

    void update_weights() {
        /*cout << "dLbydW1: \n";
        for(int i = 25; i < 30; i++) {
            cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(dLbydW1[i], 13) << " "; 
        }
        cout << "\ndLbydW2: \n";
        for(int i = 100; i < 105; i++) {
            cout << MOTION::new_fixed_point::decode<std::uint64_t, float>(dLbydW2[i], 13) << " ";
        }
        cout << endl;
        */
        dLbydW1_batches.push_back(dLbydW1);           
        dLbydW2_batches.push_back(dLbydW2);
    }

    void update_batch_weights() {
        vector<uint64_t>dLbydW1_avg(dLbydW1_batches[0].size(), (uint64_t)0);
        vector<uint64_t>dLbydW2_avg(dLbydW2_batches[0].size(), (uint64_t)0);  

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

        uint64_t rate = MOTION::new_fixed_point::encode<std::uint64_t, float>(0.05 / batch_size, 13);
        
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
        transform(output.begin(), output.end(), decoded_output.begin(), [](auto j) {
            return MOTION::new_fixed_point::decode<uint64_t, float>(j, 13);
        });
        /*cout << "Prediction: "<< endl;
        for(int i = 0; i < output.size(); i++) {
            cout << MOTION::new_fixed_point::decode<uint64_t, float>(output[i], 13) << " ";
        }*/
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
};


int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 15000;
    int batch_size = 20;
    CNN cnn = CNN();    
    cnn.initialize_random_weights({5, 845}, {5, 10}, {1, 1}, {5, 1}); //rows, columns, channels, kernels
    cnn.set_batch_size(batch_size);
    for(int i = 0; i < iterations; i++) {
        if(i % 100 == 0) cout << "\nIteration: " << i << endl;
        for(int j = 0; j < batch_size; j++) {
            cnn.get_input(i);
            cnn.get_target(i);
            cnn.forward_pass();
            cnn.backward_pass();
            cnn.update_weights();
        }
        cnn.update_batch_weights();
    }
    cout << "Testing: " << endl;
    int accuracy = 0;
    for(int i = 1; i <= 5000; i++) {
        int test_label = cnn.get_test_input(i);
        cnn.forward_pass();
        int prediction = cnn.get_prediction();
        cout << "Prediction: " << prediction << endl;
        if(prediction == test_label) accuracy++;
    }
    cout << "Accuracy: " << accuracy << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end - start);
    cout << "Time taken: " << duration.count() << " minutes" << std::endl;
    return EXIT_SUCCESS;
}