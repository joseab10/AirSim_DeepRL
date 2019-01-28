import sys
from os import path, mkdir, makedirs
SCR_PATH = path.dirname(__file__)
sys.path.append(path.join(SCR_PATH, '..','..', 'lib'))
sys.path.append(path.join(SCR_PATH, '..','..', 'cls'))
sys.path.append(path.join(SCR_PATH, '..','..', 'scr'))

import tensorflow as tf
from datetime import datetime
import drl_parser


class DRL_TB_Evaluation:

    def __init__(self, store_dir, session, config_dict):
        tf.reset_default_graph()

        self.sess = session  # tf.Session()

        if not path.exists(store_dir):
            makedirs(store_dir)

        store_dir = path.join(store_dir, "experiment_%s" % datetime.now().strftime('%y%m%d_%H%M%S'))

        self.tf_writer = tf.summary.FileWriter(store_dir, self.sess.graph)

        self._config = config_dict
        self._variables = {}

        with self.sess.graph.as_default():
            for summary in self._config['summaries']['scalar']:

                tmp_var_type = drl_parser.parse_var_type(summary['type'])
                tmp_var_name = summary['name']
                tmp_plt_name = summary['plt_name']
                tmp_var = tf.placeholder(tmp_var_type, name=tmp_var_name)
                tf.summary.scalar(tmp_plt_name, tmp_var)
                self._variables[summary['name']] = tmp_var

            self.performance_summaries = tf.summary.merge_all()


            #for summary_name, config in self._config.items():
            #    tmp_var_type = drl_parser.parse_type(config['type'])
            #    tmp_var = tf.placeholder(tmp_var_type, name=summary_name)
            #    tf.summary.scalar(config['curve_name'], tmp_var)

            #tf.reset_default_graph()
            self.sess = tf.Session()
            self.tf_writer = tf.summary.FileWriter(
                path.join(store_dir, "experiment-%s" % datetime.now().strftime("%y%m%d_%H%M%S")))


    def write_episode_data(self, episode, eval_dict):
        with self.sess.graph.as_default():
            my_dict = {}
            for var_name, var_value in eval_dict.items():
                if var_name in self._variables:
                    my_dict[self._variables[var_name]] = var_value

            summary = self.sess.run(self.performance_summaries, feed_dict=my_dict)

            self.tf_writer.add_summary(summary, episode)
            self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()






