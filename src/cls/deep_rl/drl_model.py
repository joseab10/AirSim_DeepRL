import sys
from os import path
SCR_PATH = path.dirname(__file__)
sys.path.append(path.join(SCR_PATH, '..','..', 'lib'))
sys.path.append(path.join(SCR_PATH, '..','..', 'cls'))
sys.path.append(path.join(SCR_PATH, '..','..', 'scr'))

import tensorflow as tf
import drl_parser

import json
from os import path

class DRL_Model:

    def __init__(self, model_file:str=None, config:dict=None,
                 name_suffix=''):

        if model_file is not None:
            self._init_with_file(model_file)
        elif config is not None:
            self._init_with_dict(config)
        else:
            raise NotImplementedError('No valid configuration was provided.')

        self._name_suffix = name_suffix
        self.name = self._config['name'] + self._name_suffix

        self._input_variables = {}

        # Variable_name Counters (to keep names unique)
        self._cnv_var_cnt = 0

        self._create_net()


    # Constructors (Private)
    # --------------------------------------------------------------------------------
    def _init_with_file(self, model_file):

        with open(model_file, 'r') as arq_file:
            config = json.load(arq_file)

            self._init_with_dict(config)

    def _init_with_dict(self, config:dict):

        self._config = config

    def _create_net(self):

        net_name = self._config['name'] + self._name_suffix
        net_name = net_name.upper()

        with tf.variable_scope(net_name):

            # Create Input Subnets
            subnet_outputs = []

            for subnet in self._config['subnets']:
                subnet_name = subnet['name'].upper()

                with tf.name_scope(subnet_name) and tf.variable_scope(subnet_name):
                    tmp_last_output = self._create_subnet(subnet)
                    subnet_outputs.append(tmp_last_output)

            if len(subnet_outputs) == 1:
                last_output = subnet_outputs[0]
            else:
                last_output = subnet_outputs

            # Create Output Subnet (connecting all input Subnets)
            # It is the user's responsibility to add the necessary stack/reshapes
            # layers in the config file
            self._create_output_layer(self._config['output_subnet'], last_output)

    def _create_subnet(self, config, last_output=None, output_subnet=False):

        for i, layer in enumerate(config['layers']):
            if not output_subnet and i == 0:
                if layer['type'] != 'inpt':
                    raise TypeError("First Layer of all subnets must be of type 'inpt' (" +
                                    layer['type' + ')'])
                last_output = self._create_input_layer(layer)
            else:
                last_output = self._create_hidden_layer(layer, last_output)

        return last_output

    def _create_input_layer(self, config):

        var_type = config['var type']
        var_type = drl_parser.parse_var_type(var_type)
        shape = config['shape']
        shape = self._parse_shape(shape)
        batch_dim = self._batch_dim(shape)
        name = config['name'].lower()

        with tf.name_scope(name) and tf.variable_scope(name):

            variable = tf.placeholder(var_type, shape=shape, name=name)
            self._input_variables[name] = {'var': variable, 'batch_dim': batch_dim}
            return variable

    def _create_hidden_layer(self, config, last_output):
        layer_type = config['type']

        layer_types = {
            'conv': self._create_conv_layer,
            'fuco': self._create_fuco_layer,
            'lstm': self._create_lstm_layer,
            'stck': self._create_stck_layer,
            'ustk': self._create_ustk_layer,
            'flat': self._create_flat_layer,
            'rshp': self._create_rshp_layer,
            'ccat': self._create_ccat_layer,
        }

        if layer_type not in layer_types:
            raise TypeError('Invalid Layer Type (' + layer_type + ')')

        create_layer_func = layer_types[layer_type]
        layer_name = config['name']

        with tf.name_scope(layer_name) and tf.variable_scope(layer_name):
            last_output = create_layer_func(config, last_output)

        return last_output

    def _create_output_layer(self, config, last_output):

        num_outputs = self._config['hyperparameters']['num_output_classes']
        output_net_name = config['name'].upper()

        with tf.name_scope(output_net_name) and tf.variable_scope(output_net_name):
            last_output = self._create_subnet(config,
                                              last_output=last_output,
                                              output_subnet=True)

            self.predictions = tf.layers.dense(last_output,
                                               num_outputs,
                                               name="output")

            self._create_input_layer({'name': 'actions', 'var type': 'int32', 'shape': [None]})
            self._create_input_layer({'name': 'targets', 'var type': 'float32', 'shape': [None]})

            batch_size = tf.shape(self._input_variables['actions']['var'])[
                self._input_variables['actions']['batch_dim']]
            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + \
                             self._input_variables['actions']['var']

            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Create Loss Layer
        with tf.name_scope('LOSS') and tf.variable_scope('LOSS'):
            losses = tf.squared_difference(self._input_variables['targets']['var'],
                                           self.action_predictions)

            # L2 Regularization
            l2_regularization = self._config['hyperparameters']['l2_regularization']
            l2_penalty = self._config['hyperparameters']['l2_penalty']

            if l2_regularization:
                weights = self._get_trainable_weights()
                self.l2_loss = tf.zeros([batch_size], name='l2_loss')
                for weight in weights:
                    self.l2_loss += tf.nn.l2_loss(weight)
                self.loss = tf.reduce_mean(losses + (l2_penalty * self.l2_loss))
            else:
                self.loss = tf.reduce_mean(losses)

        with tf.name_scope('OPTIMIZER') and tf.variable_scope('OPTIMIZER'):
            lr = self._config['hyperparameters']['learning_rate']
            # Optimizer Parameters from original paper
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.trainer = self.optimizer.minimize(self.loss)

    def _create_conv_layer(self, config, last_output):

        last_shape = self._layer_shape(last_output)

        name        = config['name'].lower()
        kernel_size = config['kernel size']
        filters     = config['filters']
        stride      = config['stride']
        padding     = config['padding']

        var_id = str(self._cnv_var_cnt)
        self._cnv_var_cnt += 1

        activation, initializer = drl_parser.parse_activation(config['activation'])

        kernel = tf.get_variable('kernel_' + var_id,
                                     [kernel_size, kernel_size, last_shape[-1], filters],
                                     initializer=initializer)
        bias = tf.get_variable('bias_' + var_id,
                                   [filters],
                                   initializer=initializer)
        conv_out = tf.nn.conv2d(last_output,
                                      kernel,
                                      stride,
                                      padding,
                                      name=name)
        bias_out = tf.nn.bias_add(conv_out, bias)
        last_output = activation(bias_out)

        if 'pooling' in config:
            pool_type    = config['pooling']
            pool_ksize   = config['pool ksize']
            pool_stride  = config['pool stride']
            pool_padding = config['pool padding']

            with tf.name_scope(name + '_pool'):
                    if pool_type == 'max':
                        last_output = tf.nn.max_pool(last_output,
                                                     ksize=pool_ksize,
                                                     strides=pool_stride,
                                                     padding=pool_padding,
                                                     name=name + '_pool')

        return last_output

    def _create_fuco_layer(self, config, last_output):

        name  = config['name'].lower()
        units = config['units']

        activation, _ = drl_parser.parse_activation(config['activation'])

        last_output = tf.layers.dense(last_output,
                                      units,
                                      activation=activation,
                                      name=name)

        if self._config['hyperparameters']['dropout']:
            dropout_rate = self._config['hyperparameters']['dropout_rate']
            last_output = tf.nn.dropout(last_output,
                                        dropout_rate,
                                        name=name + '_dropout')

        return last_output

    def _create_lstm_layer(self, config, last_output):

        units = config['units']

        lstm_cell = tf.nn.rnn_cell.LSTMCell(units)

        if self._config['hyperparameters']['dropout']:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=self._config['hyperparameters']['dropout_rate'])

        outputs, final_state = tf.nn.static_rnn(lstm_cell, last_output, dtype='float32')

        last_output = final_state.h

        return last_output

    def _create_stck_layer(self, config, last_outputs):

        name = config['name'].lower()
        axis = config['axis'] if 'axis' in config else 0

        last_output = tf.stack(last_outputs, axis=axis, name=name)

        return last_output

    def _create_ustk_layer(self, config, last_output):

        name = config['name'].lower()
        axis = config['axis'] if 'axis' in config else 0

        last_outputs = tf.unstack(last_output, axis=axis, name=name)
        return last_outputs

    def _create_rshp_layer(self, config, last_output):

        name  = config['name'].lower()
        shape = self._parse_shape(config['shape'])

        last_output = tf.reshape(last_output, shape, name=name)

        return last_output

    def _create_flat_layer(self, config, last_output):

        shape_list = self._layer_shape(last_output)

        flat_size = 1
        for i in range(len(shape_list)):
            if shape_list[i]:
                flat_size *= shape_list[i]

        config['shape'] = [-1, flat_size]

        last_output = self._create_rshp_layer(config, last_output)

        return last_output

    def _create_ccat_layer(self, config, last_outputs):

        name = config['name'].lower()
        axis = config['axis']

        last_output = tf.concat(last_outputs, axis, name=name)

        return last_output


    # Helper Functions (Private)
    # --------------------------------------------------------------------------------
    def _layer_shape(self, layer):
        return layer.get_shape().as_list()

    def _batch_dim(self, shape):

        batch_index = None

        for i in range(len(shape)):
            if shape[i] is None:
                batch_index = i
                break

        return batch_index

    def _get_trainable_variables(self):

        return tf.trainable_variables()

    def _get_trainable_weights(self):

        variables = self._get_trainable_variables()

        weights = []
        for variable in variables:
            if 'bias' not in variable.name:
                weights.append(variable)

        return weights

    def num_outputs(self):
        return self._config['hyperparameters']['num_output_classes']


    # Parsers (Private)
    # --------------------------------------------------------------------------------
    def _parse_shape(self, shape):

        for i in range(len(shape)):
            if shape[i] == '?':
                shape[i] = None

        return shape

    def _parse_input_dic(self, input):

        input_dict = {}
        for key, value in input.items():
            if key in self._input_variables:
                input_var = self._input_variables[key]['var']
                input_dict[input_var] = value
            else:
                raise IndexError('Variable (' + key + ') is not configured in the model.')

        return input_dict


    # Model Methods
    # --------------------------------------------------------------------------------
    def predict(self, session, inputs):

        input_dict = self._parse_input_dic(inputs)

        prediction = session.run(self.predictions, input_dict)
        return prediction

    def update(self, session, inputs):
        input_dict = self._parse_input_dic(inputs)

        _, loss = session.run([self.trainer, self.loss], input_dict)
        return loss

    def init_variables(self, session):
        with tf.name_scope('INITIALIZER') and tf.variable_scope('INITIALIZER'):
            session.run(tf.global_variables_initializer())




class DRL_TargetModel(DRL_Model):
    def __init__(self, model_file:str, config:dict=None,
                 name_suffix=''):

        DRL_Model.__init__(self, model_file=model_file, config=config, name_suffix=name_suffix + '_TARGET')
        self.tau = self._config['hyperparameters']['dqn_tau']
        self._associate = self._register_associate()

    def _register_associate(self):

        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        with tf.name_scope('SOFT_UPDATE') and tf.variable_scope('SOFT_UPDATE'):
            for idx, var in enumerate(tf_vars[0:total_vars // 2]):
                op_holder.append(tf_vars[idx + total_vars // 2].assign(
                    (var.value() * self.tau) + ((1 - self.tau) * tf_vars[idx + total_vars // 2].value())))
        return op_holder

    def update(self, session, inputs=None):
        for op in self._associate:
            session.run(op)





# Class Tests
# Uses TensorBoard to visualize the model's graph
if __name__ == '__main__':

    import argparse

    # Default arguments
    default_root_path  = path.join('..', '..', '..')
    default_model_dir  = path.join(default_root_path, 'models', 'cfg')
    default_tb_dir     = path.join(default_root_path, 'tensorboard')
    default_model_file = 'sample_net.narq.json'
    default_model_file = path.join(default_model_dir, default_model_file)

    # Argument Parser (in case this script is used to generate the visual graph
    # for some other network different from the sample one)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', action='store', default=default_model_file, help='Model Configuration Filename.')
    parser.add_argument('--tb_dir',     action='store', default=default_tb_dir,     help='Tensorboard Path.')
    args = parser.parse_args()

    model_file = args.model_file
    tb_dir     = args.tb_dir

    # Model objects
    Q = DRL_Model(model_file=model_file)
    QTarget = DRL_TargetModel(model_file=model_file)
    net_name = Q.name

    # Start tensorflow session
    sess = tf.Session()
    Q.init_variables(sess)

    # Write TensorBoard data
    tensorboard_dir = path.join(tb_dir, 'tst', net_name)

    tf_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    tf_writer.close()
    sess.close()