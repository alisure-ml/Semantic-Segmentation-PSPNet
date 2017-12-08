from __future__ import print_function
import os
import sys
from PIL import Image
import tensorflow as tf
import numpy as np

from PSPNet import PSPNet
from Tools import Tools


class Runner(object):

    def __init__(self, is_flip, num_classes, log_dir, save_dir, model_name="model.ckpt"):
        self.save_dir = Tools.new_dir(save_dir)
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        self.img_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        self.last_pool_size = 90
        self.input_size = [self.last_pool_size * 8, self.last_pool_size * 8]
        self.num_classes = num_classes

        self.is_flip = is_flip
        pass

    # 初始化网络：输入图片数据，输出预测结果
    def _init_net(self, img):
        img_shape = tf.shape(img)
        h, w = (tf.maximum(self.input_size[0], img_shape[0]), tf.maximum(self.input_size[1], img_shape[1]))
        img = self._pre_process(img, h, w)

        # Create network.
        net = PSPNet({'data': img}, is_training=False, num_classes=self.num_classes,
                     last_pool_size=self.last_pool_size)
        with tf.variable_scope('', reuse=True):
            flipped_img = tf.image.flip_left_right(tf.squeeze(img))
            flipped_img = tf.expand_dims(flipped_img, dim=0)
            net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=self.num_classes,
                          last_pool_size=self.last_pool_size)

        raw_output = net.layers["conv6"]
        if self.is_flip:
            flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
            flipped_output = tf.expand_dims(flipped_output, dim=0)
            raw_output = tf.add_n([raw_output, flipped_output])

        # Predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        predictions = tf.expand_dims(raw_output_up, dim=3)
        return predictions

    # 读取图片数据
    @staticmethod
    def _load_img(img_path):
        if os.path.isfile(img_path):
            Tools.print_info('successful load img: {0}'.format(img_path))
        else:
            Tools.print_info('not found file: {0}'.format(img_path))
            sys.exit(0)
            pass
        filename = os.path.split(img_path)[-1]
        ext = os.path.splitext(filename)[-1]
        if ext.lower() == '.png':
            img = tf.image.decode_png(tf.read_file(img_path), channels=3)
        elif ext.lower() == '.jpg':
            img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
        else:
            raise Exception('cannot process {} file.'.format(ext.lower()))
        return img, filename

    # 转换图片通道，padding图片到指定大小
    def _pre_process(self, img, h, w):
        # Convert RGB to BGR
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        img -= self.img_mean
        # padding
        pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
        pad_img = tf.expand_dims(pad_img, dim=0)
        return pad_img

    def run(self, image_path):
        # 读入图片数据
        img, filename = self._load_img(image_path)
        # 输出预测的结果
        predictions_op = self._init_net(img=img)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())

        # 加载模型
        Tools.restore_if_y(sess, self.log_dir)
        # 运行
        predictions = sess.run(predictions_op)

        # 对输出进行着色
        msk = Tools.decode_labels(predictions, num_images=1, num_classes=self.num_classes)
        Image.fromarray(msk[0]).save(os.path.join(self.save_dir, filename))
        Tools.print_info('over : result save in {}'.format(os.path.join(self.save_dir, filename)))
        pass

    pass

if __name__ == '__main__':
    Runner(is_flip=False, num_classes=19, log_dir="./model", save_dir="./output").run("data/input/test.png")
