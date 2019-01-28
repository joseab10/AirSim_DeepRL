import tensorflow as tf

def parse_var_type(var_type:str):

    var_types = {
        'int8'   : tf.int8,
        'int16'  : tf.int16,
        'int32'  : tf.int32,
        'int64'  : tf.int64,
        'uint8'  : tf.uint8,
        'uint16' : tf.uint16,
        'uint32' : tf.uint32,
        'uint64' : tf.uint64,
        'float32': tf.float32,
    }

    if var_type in var_types:
        return var_types[var_type]

    raise TypeError('Invalid Variable Type (' + var_type + ')')


def parse_activation(activation:str):

    def _actfun_linear(x):
        return x

    init_variance = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
    init_xavier   = tf.contrib.layers.xavier_initializer()
    init_normal   = tf.random_normal_initializer()

    activation_functions = {
        'relu'   : {'actfun': tf.nn.relu,     'init': init_variance},
        'sigmoid': {'actfun': tf.nn.sigmoid,  'init': init_xavier},
        'tanh'   : {'actfun': tf.nn.tanh,     'init': init_xavier},
        'linear' : {'actfun': _actfun_linear, 'init': init_normal}
    }

    if activation not in activation_functions:
        activation = 'linear'

    act_fun     = activation_functions[activation]['actfun']
    initializer = activation_functions[activation]['init']

    return act_fun, initializer