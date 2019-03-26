# -*- coding:utf-8 -*-
import h5py
import os.path
import DSTH.Common.utils
from DSTH.Common import utils

__PATH__ = '../datasets/cifar10'
batchsize=100

#计算预测的Hash码和Hash标签之间的差距的，没啥用，不用管
if __name__ == '__main__':
    # with tf.Session() as sess:
    #     hashlabelf = h5py.File(os.path.join(__PATH__, 'hashcode.hy'), 'r')
    #     hashpredictf=h5py.File(os.path.join(__PATH__, 'predicthasharray.hy'), 'r')
    #     hashlabel = hashlabelf["hashcode"].value
    #     hashpredict=hashpredictf["predicthasharray"].value
    #     print hashlabel.shape
    #     print hashpredict.shape
    #     sum=0
    #     batch_idxs=60000
    #     for i in xrange(0,60000):
    #         print hashpredict[i]
    #         print hashlabel[i]
    #         correct_predictions = sess.run(tf.equal(hashpredict[i], hashlabel[i]))
    #         accuracy = sess.run(tf.reduce_mean(tf.cast(correct_predictions, tf.float32)))
    #         sum=sum+accuracy
    #         print correct_predictions,accuracy
    #     sum=sum/60000
    #     print sum
    ids, labels, _ = utils.getidsAndimages('/tfproject/DSTH-TF/datasets/cifar10')
    predicthashstrf = h5py.File(os.path.join(__PATH__, 'predicthashstr.hy'), 'w')
    predicthashstrf.create_dataset("originlabel", data=labels)
    predicthashstrf.close()