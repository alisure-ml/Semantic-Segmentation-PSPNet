from __future__ import print_function

import os
import time

import tensorflow as tf

from PSPNet import PSPNet
from Tools import Tools
from ImageReader import ReaderTrainImageAndLabel


class Train(object):

    def __init__(self, num_classes, batch_size, input_size, log_dir, data_dir, train_list, model_name="model.ckpt"):

        # 和保存模型相关的参数
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        # 和数据相关的参数
        self.data_dir = data_dir
        self.data_train_list = train_list
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        # 随机种子，和数据增强相关
        self.random_seed = 1234

        # 和模型相关的参数：必须保证input_size大于8倍的last_pool_size
        self.last_pool_size = 90
        self.filter_number = 64

        # 和模型训练相关的参数
        self.train_beta_gamma = True
        self.weight_decay = 0.0001
        self.learning_rate = 1e-3
        self.num_steps = 40001
        self.power = 0.9
        self.update_mean_var = True
        self.momentum = 0.9

        # 读取数据
        self.reader = ReaderTrainImageAndLabel(self.data_dir, self.data_train_list, self.input_size)

        # 网络
        self.reduced_loss, self.accuracy_op, self.step_ph, self.train_op = self.build_net()

        # sess 和 saver
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        pass

    def build_net(self):
        # 读取数据
        image_batch, label_batch = self.reader.dequeue(self.batch_size)

        # 网络
        net = PSPNet({'data': image_batch}, is_training=True, num_classes=self.num_classes,
                     last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        raw_output = net.layers['conv6']

        # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
        fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv', 'conv5_3_pool3_conv',
                   'conv5_3_pool6_conv', 'conv6', 'conv5_4']
        # 所有可训练变量
        all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or
                         self.train_beta_gamma]
        # fc_list中的全连接层可训练变量和卷积可训练变量
        fc_trainable = [v for v in all_trainable if v.name.split('/')[0] in fc_list]
        conv_trainable = [v for v in all_trainable if v.name.split('/')[0] not in fc_list]  # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        # 验证
        assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(raw_output, [-1, self.num_classes])
        label_process = Tools.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]),
                                            num_classes=self.num_classes, one_hot=False)  # [batch_size, h, w]
        raw_gt = tf.reshape(label_process, [-1, ])
        # 忽略大于等于类别数的标签
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)  # [-1]
        prediction = tf.gather(raw_prediction, indices)  # [-1, num_classes]

        # 当前批次的准确率：accuracy
        accuracy = tf.equal(gt, tf.cast(tf.argmax(prediction, axis=1), tf.int32))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        # loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        l2_losses = [self.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Using Poly learning rate policy
        base_lr = tf.constant(self.learning_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / self.num_steps), self.power))

        # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
        update_ops = None if not self.update_mean_var else tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # 对变量以不同的学习率优化：分别求梯度、应用梯度
        with tf.control_dependencies(update_ops):
            # 计算梯度
            grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
            grads_conv = grads[:len(conv_trainable)]
            grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
            grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]
            # 选择优化算法
            opt_conv = tf.train.MomentumOptimizer(learning_rate, self.momentum)
            opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.momentum)
            opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.momentum)
            # 更新梯度
            train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
            train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
            train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
            # 一次完成多种操作
            train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
            pass

        return reduced_loss, accuracy, step_ph, train_op

    def train(self, save_pred_freq):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)
        tf.set_random_seed(self.random_seed)
        coord = tf.train.Coordinator()
        # 线程队列
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # Iterate over training steps.
        for step in range(self.num_steps):
            start_time = time.time()
            loss_value, accuracy, _ = self.sess.run([self.reduced_loss, self.accuracy_op, self.train_op],
                                                    feed_dict={self.step_ph: step})
            if step % save_pred_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=step)
                Tools.print_info('The checkpoint has been created.')
            duration = time.time() - start_time
            Tools.print_info('step {:d} loss={:.3f}, acc={} ({:.3f} sec/step)'.format(step, loss_value,
                                                                                      accuracy, duration))
            pass
        coord.request_stop()
        coord.join(threads)
        pass

    pass


if __name__ == '__main__':

    Train(num_classes=19, batch_size=1, input_size=[720, 720], log_dir="./model", data_dir="./data",
          train_list="data/train_list.txt").train(save_pred_freq=1000)
