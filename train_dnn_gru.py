import sys
import numpy as np
import utils.data_io_gru as data_io_gru
import tensorflow as tf
from nn.dnn_gru import DNN
from nn.sparse_autoencoder import Autoencoder

#Codes by: Yu Wang, July 2016 Copyright@MOBVOI
def TrainDNN(conf):
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print 'Reading label id...'
    label_id_file = conf.GetKey("label_id_file_path")
    (label_id, id_to_label) = data_io_gru.ReadLable(label_id_file)

    print 'Reading word lookup table...'
    word_lookup_table_file = conf.GetKey("word_lookup_table_file_path")
    (word_lookup_table, word_id, id_to_word) = data_io_gru.ReadWordLookupTable(word_lookup_table_file)
    word_lookup_table = np.asarray(word_lookup_table, dtype=np.float32)

    print 'Loading autoencoder...'
    embedding_size = len(word_lookup_table[0])
    ae_hidden_layer_size = int(conf.GetKey("ae_hidden_layer_size"))
    auto_encoder = Autoencoder(embedding_size, ae_hidden_layer_size)
    ae_param_file = conf.GetKey("ae_param_file_path") + ".npz"
    auto_encoder.LoadParam(ae_param_file)
    # output, feed_dict=auto_encoder.CompileEncodeFun(), Move it to read feature instead
    # sess.run(output,feed_dict={feed_dict:})

    reg_exp_dict = {}
    id_to_reg_exp = []
    fold = 0.2
    print 'Reading training data...'
    train_feature_file = conf.GetKey("train_feature_file_path")
    (train_data, train_ans) = data_io_gru.ReadFeature(train_feature_file, label_id, word_id, reg_exp_dict,
                                                  id_to_reg_exp, ae_hidden_layer_size, word_lookup_table, auto_encoder,
                                                  sess, 1)

    print 'Reading dev data...'
    dev_feature_file = conf.GetKey("dev_feature_file_path")
    (dev_data, dev_ans) = data_io_gru.ReadFeature(dev_feature_file, label_id, word_id, reg_exp_dict,
                                              id_to_reg_exp, ae_hidden_layer_size, word_lookup_table, auto_encoder,
                                              sess, 2)
    # 2 for testing

    hidden_layer_size = int(conf.GetKey("dnn_hidden_layer_size"))
    output_layer_size = len(label_id) + 1
    L2_reg = float(conf.GetKey("dnn_L2_reg"))
    batch_size = int(conf.GetKey("dnn_batch_size"))
    learning_rate = float(conf.GetKey("dnn_learning_rate"))
    n_epoches = int(conf.GetKey("dnn_n_epoches"))
    weights_file_dir = conf.GetKey("dnn_output_dir")
    #using K-fold for training data
    sizeData = int(fold*len(dev_data))
    sizeAns = int(fold*len(dev_ans))
    train_data=train_data+dev_data[1:sizeData]
    train_ans=train_ans+dev_ans[1:sizeAns]


    dnn = DNN(hidden_layer_size, output_layer_size, len(reg_exp_dict) + 1, ae_hidden_layer_size,
              id_to_reg_exp, id_to_word, word_lookup_table, auto_encoder, L2_reg)

    dnn.Fit(train_data, train_ans, dev_data, dev_ans, sess, batch_size, learning_rate, n_epoches, weights_file_dir)
