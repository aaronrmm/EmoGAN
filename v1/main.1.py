from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pprint import pprint
#import SharedArray as sa

from musegan.core import *
from musegan.components import *
from input_data import *
from config import *

#assign GPU


if __name__ == '__main__':

    """ Create TensorFlow Session """

    t_config = TrainingConfig

    os.environ['CUDA_VISIBLE_DEVICES'] = t_config.gpu_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_ckpt', type=str,
                        help='path to directory containing checkpoint files')
    parser.add_argument('--condition_track', type=str,
                        help='path to conditioning numpy file')
    parser.add_argument('--training_path', type=str,
                        help='path to training npz file')

    args = parser.parse_args()
    training_path =  os.path.join(os.getcwd(),'train_x_lpd_5_phr.npz') # (50266, 384, 84, 5)
    condition_track = os.path.join(os.getcwd(),'test_calm.npy')
    if args.training_path is not None:
        training_path = args.training_path
    if args.condition_track is not None:
        condition_track = args.condition_track

    training_data = np.load(training_path)
    nonzero = training_data['nonzero']
    print("starting shape "+str(nonzero.shape))#(5, 156149621)
    nonzero = nonzero.reshape(-1, 4, 48, 84, 5)
    print("reshaped shape "+str(nonzero.shape))
    nonzero = np.repeat(nonzero, 2, axis=2)
    shape = training_data['shape']#[102378, 4, 48, 84, 5] 
    print("repeated shape "+str(nonzero.shape))

    with tf.Session(config=config) as sess:

        t_config.exp_name = 'exps/nowbar_hybrid'
        model = NowbarHybrid(NowBarHybridConfig)
        input_data = InputDataNowBarHybrid(model)
        print("adding training data "+str(nonzero.shape))
        input_data.add_data_np(nonzero, 'train')
        input_data.add_data(condition_track, key='test')
        musegan = MuseGAN(sess, t_config, model)
        musegan.train(input_data)
        #musegan.dir_ckpt = 'pregen'
        #if args.dir_ckpt is not None:
        #    musegan.dir_ckpt = args.dir_ckpt
        #musegan.load(musegan.dir_ckpt)
        print("gen test")
        result, eval_result = musegan.gen_test(input_data, is_eval=True)

        save_midis(result, 'myy.midi')

