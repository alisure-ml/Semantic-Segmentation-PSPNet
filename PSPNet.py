import numpy as np
import tensorflow as tf


def layer(op):

    """Decorator for composable network layers."""
    def layer_decorated(self, *args, **kwargs):
        # 设置name参数
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))  # Automatically set a name if not provided.
        # 该层的输入terminals
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)

        # 执行被装饰的函数，得到该层的输出
        layer_output = op(self, layer_input, *args, **kwargs)  # Perform the operation and get the output.

        # 存放输出到layers中
        self.layers[name] = layer_output  # Add to layer LUT.
        # 将输出加入到临时的terminals中作为下一层的输入
        self.feed(layer_output)  # This output is now the input for the next layer.
        return self  # Return self for chained calls.

    return layer_decorated


class Network(object):

    def __init__(self, inputs, num_classes, trainable=True, is_training=False, last_pool_size=90, filter_number=64):

        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0), shape=[], name='use_dropout')
        self.is_training = is_training
        self.setup(is_training, num_classes, last_pool_size, filter_number)
        pass

    def setup(self, is_training, num_classes, last_pool_min_size, filter_number):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    # ???
    @staticmethod
    def load(data_path, session, ignore_missing=False):
        """
        Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path).item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        session.run(tf.get_variable(param_name).assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise
            pass
        pass

    # 将args从layers中获取，然后加入到terminals中
    def feed(self, *args):
        """
        Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    # 得到terminals的最后一个
    def get_output(self):
        """
        Returns the current network output.
        """
        return self.terminals[-1]

    # 得到以prefix为前缀的唯一的名字
    def get_unique_name(self, prefix):
        """
        Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        index = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, index)

    # 根据名字得到（新建）变量
    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tf.get_variable(name, shape, trainable=self.trainable)

    # def get_layer_name(self):
    #     return layer_name

    @layer
    def zero_padding(self, input, paddings, name):
        return tf.pad(input, paddings=np.array([[0, 0], [paddings, paddings], [paddings, paddings], [0, 0]]), name=name)

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding="VALID", group=1, biased=True):
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, input.get_shape()[-1], c_o])
            output = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding, data_format="NHWC")
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output
        pass

    @layer
    def atrous_conv(self, input, k_h, k_w, c_o, dilation, name, relu=True, padding="VALID", group=1, biased=True):
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, input.get_shape()[-1], c_o])
            output = tf.nn.atrous_conv2d(input, kernel, dilation, padding=padding)
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output
        pass

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding="VALID"):
        return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name,
                              data_format="NHWC")

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding="VALID"):
        return tf.nn.avg_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],padding=padding, name=name,
                              data_format="NHWC")

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = input_shape[1].value * input_shape[2].value * input_shape[-1].value
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            # Computes Relu(x * weight + biases) / Computes matmul(x, weights) + biases.
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, self.make_var('weights', shape=[dim, num_out]), self.make_var('biases', [num_out]),
                    name=scope.name)
            return fc
        pass

    @layer
    def softmax(self, input, name):
        input_shape = input.get_shape()
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                return tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                return tf.nn.softmax(input, name)
        pass

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        """
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output
        """
        with tf.variable_scope(name) as scope:
            output = tf.layers.batch_normalization(input, momentum=0.95, epsilon=1e-5, training=self.is_training,
                                                   name=name)
            if relu:
                output = tf.nn.relu(output)
            return output
        pass

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)

    pass


class PSPNet(Network):

    def setup(self, is_training, num_classes, last_pool_size, filter_number):
        """
        Network definition.
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
          last_pool_size: alisure:last pool size, default is 90
          filter_number: alisure:filter number, default is 64
        """

        (self.feed('data')
         .conv(3, 3, filter_number, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
         .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
         .relu(name='conv1_1_3x3_s2_bn_relu')
         .conv(3, 3, filter_number, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
         .batch_normalization(relu=True, name='conv1_2_3x3_bn')
         .conv(3, 3, filter_number * 2, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
         .batch_normalization(relu=True, name='conv1_3_3x3_bn')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
         .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
         .conv(1, 1, filter_number, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding1')
         .conv(3, 3, filter_number, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
         .batch_normalization(relu=True, name='conv2_1_3x3_bn')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
         .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
         .add(name='conv2_1')
         .relu(name='conv2_1/relu')
         .conv(1, 1, filter_number, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding2')
         .conv(3, 3, filter_number, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
         .batch_normalization(relu=True, name='conv2_2_3x3_bn')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
         .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
         .add(name='conv2_2')
         .relu(name='conv2_2/relu')
         .conv(1, 1, filter_number, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding3')
         .conv(3, 3, filter_number, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
         .batch_normalization(relu=True, name='conv2_3_3x3_bn')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
         .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
         .add(name='conv2_3')
         .relu(name='conv2_3/relu')
         .conv(1, 1, filter_number * 8, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
         .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')
         .conv(1, 1, filter_number * 2, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding4')
         .conv(3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
         .batch_normalization(relu=True, name='conv3_1_3x3_bn')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
         .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
         .add(name='conv3_1')
         .relu(name='conv3_1/relu')
         .conv(1, 1, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding5')
         .conv(3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
         .batch_normalization(relu=True, name='conv3_2_3x3_bn')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
         .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
         .add(name='conv3_2')
         .relu(name='conv3_2/relu')
         .conv(1, 1, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding6')
         .conv(3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
         .batch_normalization(relu=True, name='conv3_3_3x3_bn')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
         .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
         .add(name='conv3_3')
         .relu(name='conv3_3/relu')
         .conv(1, 1, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding7')
         .conv(3, 3, filter_number * 2, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
         .batch_normalization(relu=True, name='conv3_4_3x3_bn')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
         .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
         .add(name='conv3_4')
         .relu(name='conv3_4/relu')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
         .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding8')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_1_3x3')
         .batch_normalization(relu=True, name='conv4_1_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
         .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
         .add(name='conv4_1')
         .relu(name='conv4_1/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding9')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_2_3x3')
         .batch_normalization(relu=True, name='conv4_2_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
         .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
         .add(name='conv4_2')
         .relu(name='conv4_2/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding10')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_3_3x3')
         .batch_normalization(relu=True, name='conv4_3_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
         .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
         .add(name='conv4_3')
         .relu(name='conv4_3/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding11')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_4_3x3')
         .batch_normalization(relu=True, name='conv4_4_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
         .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
         .add(name='conv4_4')
         .relu(name='conv4_4/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding12')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_5_3x3')
         .batch_normalization(relu=True, name='conv4_5_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
         .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
         .add(name='conv4_5')
         .relu(name='conv4_5/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding13')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_6_3x3')
         .batch_normalization(relu=True, name='conv4_6_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
         .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
         .add(name='conv4_6')
         .relu(name='conv4_6/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_7_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_7_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding14')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_7_3x3')
         .batch_normalization(relu=True, name='conv4_7_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_7_1x1_increase')
         .batch_normalization(relu=False, name='conv4_7_1x1_increase_bn'))

        (self.feed('conv4_6/relu',
                   'conv4_7_1x1_increase_bn')
         .add(name='conv4_7')
         .relu(name='conv4_7/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_8_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_8_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding15')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_8_3x3')
         .batch_normalization(relu=True, name='conv4_8_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_8_1x1_increase')
         .batch_normalization(relu=False, name='conv4_8_1x1_increase_bn'))

        (self.feed('conv4_7/relu',
                   'conv4_8_1x1_increase_bn')
         .add(name='conv4_8')
         .relu(name='conv4_8/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_9_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_9_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding16')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_9_3x3')
         .batch_normalization(relu=True, name='conv4_9_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_9_1x1_increase')
         .batch_normalization(relu=False, name='conv4_9_1x1_increase_bn'))

        (self.feed('conv4_8/relu',
                   'conv4_9_1x1_increase_bn')
         .add(name='conv4_9')
         .relu(name='conv4_9/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_10_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_10_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding17')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_10_3x3')
         .batch_normalization(relu=True, name='conv4_10_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_10_1x1_increase')
         .batch_normalization(relu=False, name='conv4_10_1x1_increase_bn'))

        (self.feed('conv4_9/relu',
                   'conv4_10_1x1_increase_bn')
         .add(name='conv4_10')
         .relu(name='conv4_10/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_11_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_11_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding18')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_11_3x3')
         .batch_normalization(relu=True, name='conv4_11_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_11_1x1_increase')
         .batch_normalization(relu=False, name='conv4_11_1x1_increase_bn'))

        (self.feed('conv4_10/relu',
                   'conv4_11_1x1_increase_bn')
         .add(name='conv4_11')
         .relu(name='conv4_11/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_12_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_12_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding19')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_12_3x3')
         .batch_normalization(relu=True, name='conv4_12_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_12_1x1_increase')
         .batch_normalization(relu=False, name='conv4_12_1x1_increase_bn'))

        (self.feed('conv4_11/relu',
                   'conv4_12_1x1_increase_bn')
         .add(name='conv4_12')
         .relu(name='conv4_12/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_13_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_13_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding20')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_13_3x3')
         .batch_normalization(relu=True, name='conv4_13_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_13_1x1_increase')
         .batch_normalization(relu=False, name='conv4_13_1x1_increase_bn'))

        (self.feed('conv4_12/relu',
                   'conv4_13_1x1_increase_bn')
         .add(name='conv4_13')
         .relu(name='conv4_13/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_14_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_14_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding21')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_14_3x3')
         .batch_normalization(relu=True, name='conv4_14_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_14_1x1_increase')
         .batch_normalization(relu=False, name='conv4_14_1x1_increase_bn'))

        (self.feed('conv4_13/relu',
                   'conv4_14_1x1_increase_bn')
         .add(name='conv4_14')
         .relu(name='conv4_14/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_15_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_15_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding22')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_15_3x3')
         .batch_normalization(relu=True, name='conv4_15_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_15_1x1_increase')
         .batch_normalization(relu=False, name='conv4_15_1x1_increase_bn'))

        (self.feed('conv4_14/relu',
                   'conv4_15_1x1_increase_bn')
         .add(name='conv4_15')
         .relu(name='conv4_15/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_16_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_16_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding23')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_16_3x3')
         .batch_normalization(relu=True, name='conv4_16_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_16_1x1_increase')
         .batch_normalization(relu=False, name='conv4_16_1x1_increase_bn'))

        (self.feed('conv4_15/relu',
                   'conv4_16_1x1_increase_bn')
         .add(name='conv4_16')
         .relu(name='conv4_16/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_17_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_17_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding24')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_17_3x3')
         .batch_normalization(relu=True, name='conv4_17_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_17_1x1_increase')
         .batch_normalization(relu=False, name='conv4_17_1x1_increase_bn'))

        (self.feed('conv4_16/relu',
                   'conv4_17_1x1_increase_bn')
         .add(name='conv4_17')
         .relu(name='conv4_17/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_18_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_18_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding25')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_18_3x3')
         .batch_normalization(relu=True, name='conv4_18_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_18_1x1_increase')
         .batch_normalization(relu=False, name='conv4_18_1x1_increase_bn'))

        (self.feed('conv4_17/relu',
                   'conv4_18_1x1_increase_bn')
         .add(name='conv4_18')
         .relu(name='conv4_18/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_19_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_19_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding26')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_19_3x3')
         .batch_normalization(relu=True, name='conv4_19_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_19_1x1_increase')
         .batch_normalization(relu=False, name='conv4_19_1x1_increase_bn'))

        (self.feed('conv4_18/relu',
                   'conv4_19_1x1_increase_bn')
         .add(name='conv4_19')
         .relu(name='conv4_19/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_20_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_20_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding27')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_20_3x3')
         .batch_normalization(relu=True, name='conv4_20_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_20_1x1_increase')
         .batch_normalization(relu=False, name='conv4_20_1x1_increase_bn'))

        (self.feed('conv4_19/relu',
                   'conv4_20_1x1_increase_bn')
         .add(name='conv4_20')
         .relu(name='conv4_20/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_21_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_21_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding28')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_21_3x3')
         .batch_normalization(relu=True, name='conv4_21_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_21_1x1_increase')
         .batch_normalization(relu=False, name='conv4_21_1x1_increase_bn'))

        (self.feed('conv4_20/relu',
                   'conv4_21_1x1_increase_bn')
         .add(name='conv4_21')
         .relu(name='conv4_21/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_22_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_22_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding29')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_22_3x3')
         .batch_normalization(relu=True, name='conv4_22_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_22_1x1_increase')
         .batch_normalization(relu=False, name='conv4_22_1x1_increase_bn'))

        (self.feed('conv4_21/relu',
                   'conv4_22_1x1_increase_bn')
         .add(name='conv4_22')
         .relu(name='conv4_22/relu')
         .conv(1, 1, filter_number * 4, 1, 1, biased=False, relu=False, name='conv4_23_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_23_1x1_reduce_bn')
         .zero_padding(paddings=2, name='padding30')
         .atrous_conv(3, 3, filter_number * 4, 2, biased=False, relu=False, name='conv4_23_3x3')
         .batch_normalization(relu=True, name='conv4_23_3x3_bn')
         .conv(1, 1, filter_number * 16, 1, 1, biased=False, relu=False, name='conv4_23_1x1_increase')
         .batch_normalization(relu=False, name='conv4_23_1x1_increase_bn'))

        (self.feed('conv4_22/relu',
                   'conv4_23_1x1_increase_bn')
         .add(name='conv4_23')
         .relu(name='conv4_23/relu')
         .conv(1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
         .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_23/relu')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
         .zero_padding(paddings=4, name='padding31')
         .atrous_conv(3, 3, filter_number * 8, 4, biased=False, relu=False, name='conv5_1_3x3')
         .batch_normalization(relu=True, name='conv5_1_3x3_bn')
         .conv(1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
         .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
         .add(name='conv5_1')
         .relu(name='conv5_1/relu')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
         .zero_padding(paddings=4, name='padding32')
         .atrous_conv(3, 3, filter_number * 8, 4, biased=False, relu=False, name='conv5_2_3x3')
         .batch_normalization(relu=True, name='conv5_2_3x3_bn')
         .conv(1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
         .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
         .add(name='conv5_2')
         .relu(name='conv5_2/relu')
         .conv(1, 1, filter_number * 8, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
         .zero_padding(paddings=4, name='padding33')
         .atrous_conv(3, 3, filter_number * 8, 4, biased=False, relu=False, name='conv5_3_3x3')
         .batch_normalization(relu=True, name='conv5_3_3x3_bn')
         .conv(1, 1, filter_number * 32, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
         .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
         .add(name='conv5_3')
         .relu(name='conv5_3/relu'))

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        output_filter_number = filter_number * 32 // 4

        now_size = last_pool_size // 1
        (self.feed('conv5_3/relu')
         .avg_pool(now_size, now_size, now_size, now_size, name='conv5_3_pool1')
         .conv(1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
         .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        now_size = last_pool_size // 2
        (self.feed('conv5_3/relu')
         .avg_pool(now_size, now_size, now_size, now_size, name='conv5_3_pool2')
         .conv(1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
         .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        now_size = last_pool_size // 3
        (self.feed('conv5_3/relu')
         .avg_pool(now_size, now_size, now_size, now_size, name='conv5_3_pool3')
         .conv(1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
         .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        now_size = last_pool_size // 6
        (self.feed('conv5_3/relu')
         .avg_pool(now_size, now_size, now_size, now_size, name='conv5_3_pool6')
         .conv(1, 1, output_filter_number, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
         .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
         .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
         .concat(axis=-1, name='conv5_3_concat')
         .conv(3, 3, output_filter_number, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
         .batch_normalization(relu=True, name='conv5_4_bn')
         .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))

    pass