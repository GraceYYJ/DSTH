import os
import scipy.misc
import numpy as np

from Common.utils import pp, show_all_variables

import tensorflow as tf

from HashFuncLearning.dsthModel import Model
from HashFuncLearning.dsthTrain import Train

# 一些参数与超参数

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 50, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset_name", "cifar10", "The name of dataset [cifar10]")
flags.DEFINE_string("checkpoint_dir", "../dsthmodel/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    #打印上面的参数
    pp.pprint(flags.FLAGS.__flags)
    #设置输入图像的宽度与高度一致
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    #设置gpu使用率及按需慢慢增加
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        # 初始化和建立Model
        model = Model(
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            dataset_name=FLAGS.dataset_name,
            checkpoint_dir=FLAGS.checkpoint_dir,
            crop=FLAGS.crop)
        # 初始化Train类
        train = Train(sess)
        # 打印所有需要训练的变量列表
        show_all_variables()
        # 如果train参数为True，则开始训练；否则尝试读取已经训练好的model
        if FLAGS.train:
            train.train(model, FLAGS)
        else:
            if not model.load(train, FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

# run起来，你可以用这种方式直接跑的时候改参数，或者就直接run
# python dsthMain.py --epoch 50 --learning_rate 0.0002 --beta1 0.5 --batch_size 50 --input_height 32 --output_height 32 --dataset_name "cifar10" --checkpoint_dir "../checkpoint" --train True
# python dsthMain.py --train True
if __name__ == '__main__':
    tf.app.run()
