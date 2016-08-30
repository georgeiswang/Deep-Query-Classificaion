# DNN Query Classification using Tensorflow
 This file realized DNN Query Classification based on DNN and GRU RNN.
Some features in this program:
    * Using GRU RNN to understand the context of a specific word in a sentence
    * Using skip layers to learn the linear relations between regular features and output
    *

## Input information
* Train Query Size:46000+
* Test Query Size:6000+
## Parameters
    * max query length:20
    * bach size: 10
    * dropout 0.3
# #Structure and Layers
    * Input (1*300)—> GRU RNN layer —> Fully Connected layer A
    * Reg Expression (1*200)—> Fully Connected layer A
    * Input (1*300) —> Skip Layer A
    * Reg Expression (1*200) —> Skip Layer B
    * Fully Connected Layer A +Skip Layer A+ Skip Layer B—> Output Layer

# #Result
Training Accuray : 98%
Test Accuracy : 93.7%

| Method | Test Accuracy | 
|:--------:|:------------:|
| One NN| 90% | 
| Two NNs |  91.3% | 
| LSTM+NN | 93.5% |
| GRU +NN| 93.7|

##Files Explanation
    *data_io_gru.py: manipulating raw data, generating training data and test data
    *train_gru.py: Initialize the entire program
    *train_dnn_gru.py: Initialize loading data, call DNN to fit and train model 
    *dnn_gru.py: All the model structure, training and fitting process
