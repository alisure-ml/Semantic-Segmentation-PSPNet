from __future__ import print_function
import os
import time
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf

from PSPNet import PSPNet
from Tools import Tools
from ImageReader import ReaderTestImage


class Runner(object):

    def __init__(self, batch_size, is_flip, num_classes, test_list, log_dir, save_dir, model_name="model.ckpt"):
        self.test_list = test_list
        self.save_dir = Tools.new_dir(save_dir)
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.last_pool_size = 90
        self.filter_number = 64

        self.is_flip = is_flip

        # 读取数据
        self.reader = ReaderTestImage(self.test_list)
        # net
        self.raw_output_op, self.images_name_op, self.images_size_op = self._init_net()

        # sess
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        pass

    # 初始化网络：输入图片数据，输出预测结果
    def _init_net(self):
        images, images_flip, images_name, image_size = self.reader.dequeue(num_elements=self.batch_size)
        # Create network.
        net = PSPNet({'data': images}, is_training=False, num_classes=self.num_classes,
                     last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        raw_output = net.layers["conv6"]
        # 翻转
        if self.is_flip:
            with tf.variable_scope('', reuse=True):
                net2 = PSPNet({'data': images_flip}, is_training=False, num_classes=self.num_classes,
                              last_pool_size=self.last_pool_size, filter_number=self.filter_number)
            flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
            flipped_output = tf.expand_dims(flipped_output, axis=0)
            raw_output = tf.add_n([raw_output, flipped_output])
        return raw_output, images_name, image_size

    # 着色
    def _save_result(self, predictions, duration, file_names):
        for index, file_name in enumerate(file_names):
            file_name = os.path.split(str(file_name).split("'")[1])[1]
            Image.fromarray(predictions[index]).convert("RGB").save(os.path.join(self.save_dir, file_name))
            Tools.print_info('duration {} save in {}'.format(duration, os.path.join(self.save_dir, file_name)))
        pass

    def test(self):
        # 加载模型
        Tools.restore_if_y(self.sess, self.log_dir)
        # 运行
        coord = tf.train.Coordinator()
        # 线程队列
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # Iterate over training steps.
        batch_number = np.ceil(len(self.reader.images_list)/self.batch_size)
        for step in range(int(batch_number)):
            start_time = time.time()
            raw_outputs, images_name, images_size = self.sess.run([self.raw_output_op, self.images_name_op,
                                                                   self.images_size_op])
            # Predictions：图片变成原来的大小
            predictions = []
            for index, raw_output in enumerate(raw_outputs):
                raw_output = tf.expand_dims(raw_output, axis=0)
                raw_output = tf.image.resize_bilinear(raw_output, size=images_size[index], align_corners=True)
                raw_output = tf.argmax(raw_output, axis=3)
                prediction = tf.expand_dims(raw_output, dim=3)
                prediction = tf.squeeze(prediction, axis=0)
                predictions.append(self.sess.run(prediction))
            duration = time.time() - start_time
            # 着色
            images_result = Tools.decode_labels_test(predictions, num_images=1, num_classes=self.num_classes)
            self._save_result(images_result, duration, images_name)
            pass
        coord.request_stop()
        coord.join(threads)
        pass

    pass

if __name__ == '__main__':

    Runner(is_flip=True, num_classes=19, batch_size=1,
           test_list="data/input/test_list.txt", log_dir="./model", save_dir="./output").test()
