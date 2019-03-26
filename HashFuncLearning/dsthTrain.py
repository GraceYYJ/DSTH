from __future__ import division

import time
from Common.utils import getidsAndimages, getHashtags
from .ops import *
from .dsthModel import *

from Common.util import *


class Train(object):
    def __init__(self, sess):
        self.sess = sess

    def train(self, model, config):
        # 定义梯度下降优化器
        n_optim = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(model.loss, var_list=model.t_vars)
        # n_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(model.loss, var_list=model.t_vars)

        # tf初始化所有变量，1.4用try里面的
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        # 写log和画图的，忽略
        self.writer = SummaryWriter("../logs", self.sess.graph)
        self.n_sum = merge_summary([model.loss_sum])
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = model.load(self, model.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        #获取图像数据与Hash标签
        ids, datas, self.data = getidsAndimages(config.dataset_name)
        self.hashtags = getHashtags(config.dataset_name).astype(np.int32)

        #训练
        batch_idxs = len(ids) // config.batch_size
        for epoch in xrange(config.epoch):
            for idx in xrange(0, batch_idxs):
                # 这一批次的图像数据和Hash标签
                batch1 = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_images = np.array(batch1).astype(np.float32)
                batch2 = self.hashtags[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_hashtags = np.array(batch2).astype(np.float32)
                # tensorboard画图
                _, summary_str = self.sess.run([n_optim, self.n_sum],
                                               feed_dict={model.inputs: batch_images,
                                                          model.hashtags: batch_hashtags})
                self.writer.add_summary(summary_str, counter)
                # 网络输出
                logits = model.logits.eval({model.inputs: batch_images,
                                            model.hashtags: batch_hashtags})
                err = model.loss.eval({model.inputs: batch_images,
                                       model.hashtags: batch_hashtags})
                accracy = model.accuracy.eval({model.inputs: batch_images,
                                               model.hashtags: batch_hashtags})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accracy:%8f" \
                      % (epoch, idx, batch_idxs, time.time() - start_time, err, accracy))
                print("logit:")
                print(logits)
                print("hashtags")
                print(batch_hashtags)
                # 保存模型参数 meta文件那些
                if np.mod(counter, 500) == 2:
                    model.save(self, config.checkpoint_dir, counter)
