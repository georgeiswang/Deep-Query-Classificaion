import sys
from utils.config_reader import ConfigReader
from train_dnn_gru import TrainDNN
#from train_autoencoder import TrainAutoencoder

#Codes by: Yu Wang, July 2016 Copyright@MOBVOI
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python train_gru.py config_file'
        exit(1)
    config_file = sys.argv[1]
    conf = ConfigReader(config_file)
    conf.ReadConfig()
    #TrainAutoencoder(conf)
    TrainDNN(conf)
