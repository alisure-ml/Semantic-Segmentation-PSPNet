import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf


class Tools(object):

    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_info(info):
        print("{} {}".format(time.strftime("%H:%M:%S", time.localtime()), info))
        pass

    @staticmethod
    def to_txt(data, file_name):
        with open(file_name, "w") as f:
            for one_data in data:
                f.write("{}\n".format(one_data))
            pass
        pass

    # 对输出进行着色
    @staticmethod
    def decode_labels(mask, num_images, num_classes):
        # 0 = road, 1 = sidewalk, 2 = building, 3 = wall, 4 = fence, 5 = pole,
        # 6 = traffic light, 7 = traffic sign, 8 = vegetation, 9 = terrain, 10 = sky,
        # 11 = person, 12 = rider, 13 = car, 14 = truck, 15 = bus,
        # 16 = train, 17 = motocycle, 18 = bicycle, 19 = void label
        label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69), (102, 102, 156), (190, 153, 153),
                         (153, 153, 153), (250, 170, 29), (219, 219, 0), (106, 142, 35), (152, 250, 152),
                         (69, 129, 180), (219, 19, 60), (255, 0, 0), (0, 0, 142), (0, 0, 69),
                         (0, 60, 100), (0, 79, 100), (0, 0, 230), (119, 10, 32), (1, 1, 1)]

        n, h, w, c = mask.shape
        
        assert (n >= num_images), \
            'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        return outputs

    # 对输出进行着色
    @staticmethod
    def decode_labels_test(mask, num_images, num_classes):
        # 0 = road, 1 = sidewalk, 2 = building, 3 = wall, 4 = fence, 5 = pole,
        # 6 = traffic light, 7 = traffic sign, 8 = vegetation, 9 = terrain, 10 = sky,
        # 11 = person, 12 = rider, 13 = car, 14 = truck, 15 = bus,
        # 16 = train, 17 = motocycle, 18 = bicycle, 19 = void label
        label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69), (102, 102, 156), (190, 153, 153),
                         (153, 153, 153), (250, 170, 29), (219, 219, 0), (106, 142, 35), (152, 250, 152),
                         (69, 129, 180), (219, 19, 60), (255, 0, 0), (0, 0, 142), (0, 0, 69),
                         (0, 60, 100), (0, 79, 100), (0, 0, 230), (119, 10, 32), (1, 1, 1)]
        mask = np.array(mask)
        n, h, w, c = mask.shape

        assert (n >= num_images), \
            'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        return outputs

    @staticmethod
    def prepare_label(input_batch, new_size, num_classes, one_hot=True):
        with tf.name_scope('label_encode'):
            # as labels are integer numbers, need to use NN interp.
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3])  # reducing the channel dimension.
            if one_hot:
                input_batch = tf.one_hot(input_batch, depth=num_classes)
        return input_batch

    # 如果模型存在，恢复模型
    @staticmethod
    def restore_if_y(sess, log_dir):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver(var_list=tf.global_variables()).restore(sess, ckpt.model_checkpoint_path)
            Tools.print_info("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            Tools.print_info('No checkpoint file found.')
            pass
        pass

    pass
