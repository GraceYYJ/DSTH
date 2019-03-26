# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
import datetime
import numpy as np
import h5py
import os.path

import sys

sys.path.append('/tfproject/FDDA')
from Common.util import Utils

__PATH__MODEL = '../dsthmodel/checkpoint/cifar10_50_32_32/'
__PATH__DATA = '../datasets/cifar10/'
__NAME__DATA = 'cifar10'

batchsize = 100

if __name__ == '__main__':
    with tf.Session() as sess:
        # resotore 训练好的模型
        saver = tf.train.import_meta_graph(__PATH__MODEL + 'DSTH.model-59502.meta')
        saver.restore(sess, tf.train.latest_checkpoint(__PATH__MODEL))
        graph = tf.get_default_graph()
        #打印出模型里保存的所有玩意儿
        for op in graph.get_operations():
            print(op.name, op.values())

        features = []
        starttime = datetime.datetime.now()
        #根据之前建立模型给每个图节点定义的名字，用graph把这个节点拿出来
        inputs = graph.get_tensor_by_name('images:0')
        #这一个slice15就是最后降维后的48位feature的节点
        slice15 = graph.get_tensor_by_name('network32/n_slice/concat_14:0')
        #获取数据集图像
        ids, labels, images = Utils.getidsAndimages(__PATH__DATA)
        #分batch获取48位的features
        batch_idxs = len(ids) // batchsize
        for idx in range(0, batch_idxs):
            batch = images[idx * batchsize:(idx + 1) * batchsize]
            batch_images = np.array(batch).astype(np.float32)
            #根据传进去这一批的images获取这一批48位的features
            features48 = sess.run(slice15, feed_dict={inputs: batch_images})
            print(features48)  # [[ 0.]] Yay!
            features.extend(features48)
        #这就是所有数据的48位feature了
        features = np.asarray(features, dtype=np.float32)
        #print(features)
        print(features.shape)
        #存起来
        predictfeatures48 = h5py.File(os.path.join(__PATH__DATA, 'features48.hy'), 'w')
        predictfeatures48.create_dataset("features48", data=features)
        predictfeatures48.close()

        print("finish")
        endtime = datetime.datetime.now()
        usetime = (endtime - starttime).seconds
        print(usetime, "seconds")
