import theano
from convnet.icdar2013 import ICDAR2013
from pylearn2.config import yaml_parse

train = open('conv.yaml', 'r').read()
train_params = {'save_path': '.'}
train = train % train_params

train = yaml_parse.load(train)
train.main_loop()

# ICDAR2013(data_path="../datasets/icdar2013/",
#           which_set="cropped_images",
#           groundtruth="cropped_labels", max_count=10)