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

    args = parser.parse_args()
    training_path =  'C://Users//Aaron//Desktop//school//generative_methods//EmoGAN//train_x_lpd_5_phr.npz' # (50266, 384, 84, 5)
    training_data = np.load(training_path)
    nonzero = training_data['nonzero']
    shape = training_data['shape']
    from pprint import pprint
    pprint(vars(training_data))
    f = training_data.f
    #training_data = np.fromfile(training_path, dtype=np.float32)
    #print("training data shape")
   # print(str(training_data.shape))
    with tf.Session(config=config) as sess:

        t_config.exp_name = 'exps/nowbar_hybrid'
        model = NowbarHybrid(NowBarHybridConfig)
        input_data = InputDataNowBarHybrid(model)
        #input_data.add_data(path_x_train_phr, 'train')
        condition_track = 'C://Users//Aaron//Desktop//school//test_calm.npy'
        if args.condition_track is not None:
            condition_track = args.condition_track
        print("adding input data")
        input_data.add_data(condition_track, key='test')
        musegan = MuseGAN(sess, t_config, model)
        input_data.add_data_np(nonzero, 'train')
        musegan.train(input_data)
        #musegan.dir_ckpt = 'C://Users//Aaron//Desktop//school//generative_methods//EmoGAN//lastfm_alternative_g_composer_d_proposed'
        #if args.dir_ckpt is not None:
        #    musegan.dir_ckpt = args.dir_ckpt
        #musegan.load(musegan.dir_ckpt)
        print("gen test")
        result, eval_result = musegan.gen_test(input_data, is_eval=True)

        save_midis(result, 'C://Users//Aaron//Desktop//myy.midi')

