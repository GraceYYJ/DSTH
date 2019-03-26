from __future__ import division
import os

from .ops import *
from Common.utils import *


class Model(object):
    # 初始化网络
    def __init__(self, input_height=32, input_width=32, output_height=32,
                 output_width=32, batch_size=64, f1_dim=32, f2_dim=32,
                 f3_dim=64, f4_dim=64, f5_dim=128, f6_dim=128, fc_dim=4096,
                 hashbit=48, slicenum=16, outbit=3, size=32, c_dim=3,
                 dataset_name='default', checkpoint_dir="../dsthmodel/checkpoint", crop=False):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          f1_dim: (optional) Dimension of gen filters in first conv layer. [64]
          f2_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          f3_dim: (optional) Dimension of gen filters in first conv layer. [64]
          f4_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          f5_dim: (optional) Dimension of gen filters in first conv layer. [64]
          f6_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.f1_dim = f1_dim
        self.f2_dim = f2_dim
        self.f3_dim = f3_dim
        self.f4_dim = f4_dim
        self.f5_dim = f5_dim
        self.f6_dim = f6_dim
        self.fc_dim = fc_dim
        self.c_dim = c_dim
        self.crop = crop

        self.bn3 = batch_norm(name='bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.hashbit = hashbit
        self.slicenum = slicenum
        self.outbit = outbit
        self.size = size
        if self.dataset_name == 'cifar10':
            self.c_dim = 3

        self.build_model()

    # 建立模型
    def build_model(self):
        #这一坨没啥用。。。原本是为了判断是否需要对原始图像crop的，忽略吧，反正用的都是cifar 32*32的图
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        # 输入图像和Hash标签
        self.inputs = tf.placeholder(tf.float32, name='images', shape=[None] + image_dims)
        self.hashtags = tf.placeholder(tf.float32, name='hashbit', shape=[None, self.hashbit])
        inputs = self.inputs
        hashtags = self.hashtags

        # 网络关键部分就在network，logits是输出的未二值化的值，32代表的是这是32*32的图像，当初写的时候以防要送入其他大小的图像为了区别网络写的
        self.logits = self.network(inputs, 32)

        # 损失函数定义
        def sqrt_l2_loss_2(x, y):
            diff = tf.subtract(y, x)
            loss = tf.sqrt(tf.reduce_sum(tf.square(diff), 1))
            return loss
        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(sqrt_l2_loss_2(self.logits, hashtags))

        # tensorboard画图用的，忽略
        self.loss_sum = scalar_summary("loss", self.loss)

        # abslogits是对网络输出求绝对值，predictions是以0.5为界限进行二值化，也就是Hash码得到的地方
        with tf.variable_scope('Accuracy'):
            abslogits = tf.abs(self.logits, name="abslogits")
            predictions = tf.cast(tf.greater(abslogits, 0.5), tf.int32, name="predictions")
            #这是没事做对比一下hash码和hash标签用的，计算一下网络对hash标签学习的精确性，忽略
            hashtags = tf.cast(hashtags, tf.int32)
            correct_predictions = tf.equal(predictions, hashtags, name="correct_predictions")
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    # 这是核心网络，3*（卷积+池化）+fc+slice，slice部分具体进入到sliceop里，比较复杂，我已经失忆了
    def network(self, image, size):
        with tf.variable_scope("network" + str(size)) as scope:
            conv1 = conv2d(image, self.f1_dim, name='n_conv1')
            pool1 = max_pool(conv1, name='n_pool1')
            conv2 = conv2d(pool1, self.f2_dim, name='n_conv2')
            pool2 = avg_pool(conv2, name='n_pool2')
            conv3 = tf.nn.relu(self.bn3(conv2d(pool2, self.f3_dim, name='n_conv3')), name='n_relu3')
            pool3 = avg_pool(conv3, name='n_pool3')
            fc = linear(tf.reshape(pool3, [-1, 3 * 3 * 64]), 4096, name='n_fc')
            slices = sliceop(fc, self.slicenum, self.outbit, name='n_slice')
            return slices

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, train, checkpoint_dir, step):
        model_name = "DSTH.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(train.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, train, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(train.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            # print (train.sess.run(self.logits))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
