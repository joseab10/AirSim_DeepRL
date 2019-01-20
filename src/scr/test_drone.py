from __future__ import print_function

import gym
from dqn.dqn_agent import DQNAgent

from dqn.networks import *
import numpy as np

import os
import json
from datetime import datetime

import argparse

np.random.seed(0)


def test_model(model_path, model_suffix = "", hist_len = 3,
               n_test_episodes=15,
               conv_layers=2, fc_layers=1, verbose=True):
    env = gym.make("CarRacing-v0").unwrapped

    img_width = 96
    img_height = 96
    num_actions = 5
    lr = 1e-4
    tau = 0.01

    Q = CNN(img_width, img_height, hist_len + 1, num_actions, lr,
            conv_layers=conv_layers, fc_layers=fc_layers)
    Q_Target = CNNTargetNetwork(img_width, img_height, hist_len + 1, num_actions, lr, tau,
                                conv_layers=conv_layers, fc_layers=fc_layers)

    agent = DQNAgent(Q, Q_Target, num_actions)

    agent.load(model_path + '/dqn_agent.ckpt')

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, do_prefill=False, rendering=True, history_length=hist_len, verbose=verbose)
        episode_rewards.append(stats.episode_reward)
        print("Episode", " Reward:", stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s" % datetime.now().strftime("%Y%m%d-%H%M%S") + model_suffix + ".json"
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')



if __name__ == "__main__":

    from train_carracing import run_episode

    history_length = 3


    parser = argparse.ArgumentParser()

    parser.add_argument('--model_suffix', action='store', default='',
                        help='Name suffix to identify models, data and results.', type=str)

    parser.add_argument('--his_len', action='store', default=4,
                        help='History Length for CNN.', type=int)

    parser.add_argument('--conv_lay', action='store', default=2,
                        help='Number of Convolutional Layers.', type=int)

    parser.add_argument('--fc_lay', action='store', default=1,
                        help='Number of Fully Connected Layers.', type=int)

    parser.add_argument('--n_episodes', action='store', default=15,
                        help='Number Testing Episodes.', type=int)


    args = parser.parse_args()

    name_suffix = args.model_suffix
    model_dir = "./models/carracing/"

    if name_suffix is not "":
        model_dir  += name_suffix
        name_suffix = "_"  + name_suffix

    hist_len = args.his_len
    conv_layers = args.conv_lay
    fc_layers = args.fc_lay
    n_episodes = args.n_episodes


    test_model(model_dir, name_suffix, hist_len=hist_len, conv_layers=conv_layers, fc_layers=fc_layers,
               n_test_episodes=n_episodes, verbose=True)
