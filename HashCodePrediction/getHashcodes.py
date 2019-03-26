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

        predicthashcode = []
        predicthashstr = []
        starttime = datetime.datetime.now()
        # 根据之前建立模型给每个图节点定义的名字，用graph把这个节点拿出来
        inputs = graph.get_tensor_by_name('images:0')
        # 这一个predictions就是最后48位feature二值化后的Hash码值
        predictions = graph.get_tensor_by_name('Accuracy/predictions:0')
        # 获取数据集图像
        ids, labels, images = Utils.getidsAndimages(__NAME__DATA)
        batch_idxs = len(ids) // batchsize
        # 分batch获取Hash码
        for idx in range(0, batch_idxs):
            batch = images[idx * batchsize:(idx + 1) * batchsize]
            batch_images = np.array(batch).astype(np.float32)
            # 根据传进去这一批的images获取这一批的Hash码
            hashcode = sess.run(predictions, feed_dict={inputs: batch_images})
            print(hashcode)  # [[ 0.]] Yay!
            predicthashcode.extend(hashcode)
        # 这就是所有数据的Hash码了
        predicthashcode = np.asarray(predicthashcode, dtype=np.int32)
        print('predicthashcode:', predicthashcode.shape)
        # 存起来
        predicthashs = h5py.File(os.path.join(__PATH__DATA, 'predicthasharray.hy'), 'w')
        predicthashs.create_dataset("predicthasharray", data=predicthashcode)
        predicthashs.close()
        # 上面是n个[1,0,0,1,...]这样的Hash码，你要是想要"1001..."这样字符串形式的，下面这个帮你转一下
        # 这一大段代码都很傻x，毕竟年轻瞎写
        for i in range(len(predicthashcode)):
            strx = "".join(str(j) for j in predicthashcode[i])
            predicthashstr.append(strx.encode())
        # print(predicthashstr)
        predicthashstr = np.asarray(predicthashstr)
        predicthashstrf = h5py.File(os.path.join(__PATH__DATA, 'predicthashstr.hy'), 'w')
        predicthashstrf.create_dataset("predicthashstr", data=predicthashstr)
        predicthashstrf.close()

    print("finish")
    endtime = datetime.datetime.now()
    usetime = (endtime - starttime).seconds
    print(usetime, "seconds")
