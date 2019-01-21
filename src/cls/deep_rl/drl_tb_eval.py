import sys
from os import path
SCR_PATH = path.dirname(__file__)
sys.path.append(path.join(SCR_PATH, '..','..', 'lib'))
sys.path.append(path.join(SCR_PATH, '..','..', 'cls'))
sys.path.append(path.join(SCR_PATH, '..','..', 'scr'))

import tensorflow as tf
import datetime
import drl_parser


class DRL_TB_Evaluation:

    def __init__(self, store_dir, session, config_dict):
        tf.reset_default_graph()

        self.sess = session  # tf.Session()

        store_dir = path.join(store_dir, "experiment_%s" % datetime.now().strftime('%y%m%d_%H%M%S'))

        self.tf_writer = tf.summary.FileWriter(store_dir, self.sess.graph)

        self._config = config_dict

        with self.sess.graph.as_default():
            for summary in self._config['summaries']['scalar']:

                tmp_var_type = drl_parser.parse_type(summary['type'])
                tmp_var = tf.placeholder(tmp_var_type, name=summary['name'])
                tf.summary.scalar(summary['plot_title'], tmp_var)

            self.performance_summaries = tf.summary.merge_all()


            for summary_name, config in self._config.items():
                tmp_var_type = drl_parser.parse_type(config['type'])
                tmp_var = tf.placeholder(tmp_var_type, name=summary_name)
                tf.summary.scalar(config['curve_name'], tmp_var)

            tf.reset_default_graph()
            self.sess = tf.Session()
            self.tf_writer = tf.summary.FileWriter(
                path.join(store_dir, "experiment-%s" % datetime.now().strftime("%y%m%d_%H%M%S")))


    def write_episode_data(self, episode, eval_dict):
        with self.sess.graph.as_default():
            my_dict = {}
            for k in eval_dict:
                assert (k in self.stats)
                my_dict[self.pl_stats[k]] = eval_dict[k]

            summary = self.sess.run(self.performance_summaries, feed_dict=my_dict)

            self.tf_writer.add_summary(summary, episode)
            self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()






