# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from gym import wrappers

from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import *

import argparse

from schedule import Schedule
from early_stop import EarlyStop




def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, do_prefill=False,
                rendering=False, max_timesteps=1000, history_length=0, verbose=False):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0

    state = env.reset()
    #env._max_episode_steps = max_timesteps

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act([state], deterministic)
        action = id_to_action(action_id)

        if verbose:
            print('\tStep ', '{:7d}'.format(step), ' Action: ', ACTIONS[action_id]['log'])

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        if do_prefill:
            agent.replay_buffer.add_transition(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        if step % 100 == 0 and False:
            print('\t\tStep ', '{:4d}'.format(step), ' Reward: ', '{:4.4f}'.format(stats.episode_reward))

        step += 1

    return stats


def train_online(env, agent, num_episodes, epsilon_schedule, early_stop,
                 history_length=0, max_timesteps=1000,
                 model_dir="./models_carracing", tensorboard_dir="./tensorboard", tensorboard_suffix="", rendering=False,
                 skip_frames=0):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "validation_reward",
                                                                      "episode_reward_100", "validation_reward_10",
                                                                      "episode_duration", "epsilon",
                                                                      "straight", "left", "right", "accel", "brake"],
                             dir_suffix=tensorboard_suffix)

    valid_reward = 0

    # Averaged (Smoothed) reward measurements
    train_rewards_100 = np.zeros(100)
    valid_rewards_10 = np.zeros(10)
    train_reward_100 = 0
    valid_reward_10 = 0

    for i in range(num_episodes):
        #print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        deterministic = False
        training = True
        do_rendering = rendering

        # Validation (Deterministic)
        if i % 10 == 0:
            deterministic = True
            #if i % 100 == 0:
            #    do_rendering = True

        if epsilon_schedule is not None:
            agent.epsilon = epsilon_schedule(i)
       
        stats = run_episode(env, agent, max_timesteps=max_timesteps,
                            deterministic=deterministic, do_training=training,
                            rendering=do_rendering, history_length=history_length, skip_frames=skip_frames)

        ep_type = '   '
        if i % 10 == 0:
            valid_reward = stats.episode_reward
            ep_type = '(v)'

            valid_rewards_10 = np.append(valid_rewards_10, valid_reward)
            valid_reward_10 += (valid_reward - valid_rewards_10[0]) / 10
            valid_rewards_10 = valid_rewards_10[1:]

        train_rewards_100 = np.append(train_rewards_100, stats.episode_reward)
        train_reward_100 += (stats.episode_reward - train_rewards_100[0]) / 100
        train_rewards_100 = train_rewards_100[1:]

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward"      : stats.episode_reward,
                                                      "validation_reward"   : valid_reward,
                                                      "episode_reward_100"  : train_reward_100,
                                                      "validation_reward_10": valid_reward_10,
                                                      "episode_duration"    : stats.episode_steps,
                                                      "straight"            : stats.get_action_usage(STRAIGHT),
                                                      "left"                : stats.get_action_usage(LEFT),
                                                      "right"               : stats.get_action_usage(RIGHT),
                                                      "accel"               : stats.get_action_usage(ACCELERATE),
                                                      "brake"               : stats.get_action_usage(BRAKE),
                                                      "epsilon"             : agent.epsilon
                                                      })

        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

        print('Episode ', ep_type, ': ', '{:7d}'.format(i), ' Reward: ', '{:4.4f}'.format(stats.episode_reward))

        # Early Stopping
        early_stop.step(valid_reward_10)
        if early_stop.save_flag:
            if early_stop.stop:
                break
            else:
                agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

def prefill_buffer(env, agent, rendering = False, max_timesteps = 1000, history_length=0):

    episode = 0
    while (not agent.replay_buffer.has_min_items()):

        episode += 1
        stats = run_episode(env, agent, deterministic=False, do_training=False, do_prefill=True,
                            max_timesteps=max_timesteps, history_length=history_length, rendering=rendering)

        print("Prefill Episode: ", '{:7d}'.format(episode), ' Reward: ', '{:4.0f}'.format(stats.episode_reward),
              ' Buffer Filled: ', '{:8d}'.format(agent.replay_buffer.len()))



if __name__ == "__main__":

    from test_carracing import test_model

    parser = argparse.ArgumentParser()

    parser.add_argument('--his_len', action='store',      default=4,
                        help='History Length for CNN.',       type=int)
    parser.add_argument('--lr',      action='store',      default=1e-4,
                        help='Learning Rate.',                type=float)
    parser.add_argument('--tau',     action='store',      default=0.01,
                        help='Soft-Update Interpolation Parameter (Tau).', type=float)
    parser.add_argument('--df',      action='store',      default=0.99,
                        help='Past Rewards Discount Factor.', type=float)
    parser.add_argument('--bs',      action='store',      default=100,
                        help='Batch Size.',                   type=int)
    parser.add_argument('--episodes',action='store',      default=10000,
                        help='Maximum Number of Training Episodes.', type=int)
    parser.add_argument('--skip_frames', action='store', default=3,
                        help='Skip Frames during training.', type=int)

    # Model
    parser.add_argument('--conv_lay', action='store', default=2,
                        help='Number of Convolutional Layers.', type=int)
    parser.add_argument('--fc_lay', action='store', default=1,
                        help='Number of Fully Connected Layers.', type=int)

    parser.add_argument('--ddqn', action='store_true', default=False,
                        help='Use Double-DQN.')
    parser.add_argument('--model_suffix', action='store', default='',
                        help='Name suffix to identify models, data and results.', type=str)

    # Epsilon
    parser.add_argument('--e_0',     action='store',      default=0.75,
                        help='Initial Random Exploration rate (Epsilon).', type=float)
    parser.add_argument('--e_min',   action='store',      default=0.05,
                        help='Minimum Exploration Rate.'     , type=float)
    parser.add_argument('--e_df', action='store', default='exponential',
                        help='Random Exploration Decay Function.', type=str)
    parser.add_argument('--e_steps', action='store', default=150,
                        help='Random Exploration Decay Episodes.', type=int)
    parser.add_argument('--e_ann', action='store_true', default=False,
                        help='Random Exploration Decay Cosine Annealing.')
    parser.add_argument('--e_acyc', action='store', default=10,
                        help='Random Exploration Decay Cosine Annealing Cycles.', type=int)

    # Early Stop
    parser.add_argument('--patience', action='store', default=25,
                        help='Early Stop Stalled Episodes Patience.', type=int)

    parser.add_argument('--render',  action='store_true', default=False,
                        help='Render Environment.'                      )

    args = parser.parse_args()

    img_width = 96
    img_height = 96
    num_actions = 5

    # Learning Rate
    lr          = args.lr
    # Soft-Update Interpolation Parameter
    tau         = args.tau

    hist_len    = args.his_len

    # Past Rewards Discount Factor (Gamma)
    discount_factor = args.df

    batch_size = args.bs

    # Random Exploration Rate (Epsilon)
    epsilon0         = args.e_0
    min_epsilon      = args.e_min
    decay_function   = args.e_df
    cosine_annealing = args.e_ann
    annealing_cycles = args.e_acyc
    decay_episodes   = args.e_steps

    # Model
    conv_layers = args.conv_lay
    fc_layers = args.fc_lay

    name_suffix = args.model_suffix
    model_dir = "./models/carracing/"
    tensorboard_dir = './tensorboard/carracing'
    log_dir = "./log"
    log_file = 'training'

    if name_suffix is not "":
        model_dir  += name_suffix
        name_suffix = "_"  + name_suffix
        log_file += name_suffix

    log_file += datetime.now().strftime("%Y%m%d-%H%M%S") + '.out'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    sys.stdout = open(log_dir + "/" +  log_file, 'w')

    # Early Stop
    early_stop_patience = args.patience

    # Render the carracing 2D environment
    rendering   = args.render

    max_timesteps = 1000

    # Maximum number of training Episodes
    num_episodes = args.episodes

    skip_frames = args.skip_frames

    double_dqn = args.ddqn

    buffer_capacity = 100000

    env = gym.make('CarRacing-v0').unwrapped

    # Random Action Probability Distribution
    act_probabilities = np.ones(num_actions)
    #act_probabilities[STRAIGHT]   = 5
    #act_probabilities[ACCELERATE] = 40
    #act_probabilities[LEFT]       = 30
    #act_probabilities[RIGHT]      = 30
    #act_probabilities[BRAKE]      = 1

    # According to Tensorboard's best training session (in %)
    act_probabilities[STRAIGHT] = 18.05
    act_probabilities[ACCELERATE] = 40.55
    act_probabilities[LEFT] = 19.66
    act_probabilities[RIGHT] = 18.05
    act_probabilities[BRAKE] = 3.687

    act_probabilities /= np.sum(act_probabilities)


    Q = CNN(img_width, img_height, hist_len + 1, num_actions, lr,
            conv_layers=conv_layers, fc_layers=fc_layers)
    Q_Target = CNNTargetNetwork(img_width, img_height, hist_len + 1, num_actions, lr, tau,
                                conv_layers = conv_layers, fc_layers = fc_layers)

    # Start with epsilon=1 for the buffering, so that all actions are random and in the specified probabilities
    # instead of randomly depending on the initialized parameters
    agent = DQNAgent(Q, Q_Target, num_actions,
                     discount_factor=discount_factor, batch_size=batch_size,
                     epsilon=1, act_probabilities=act_probabilities,
                     double_q=double_dqn, buffer_capacity=buffer_capacity, prefill_bs_percentage=10)

    # Exploration-vs-Exploitation Parameter (Epsilon) Schedule
    epsilon_schedule = Schedule(epsilon0, min_epsilon, decay_episodes, decay_function=decay_function,
                                cosine_annealing=cosine_annealing, annealing_cycles=annealing_cycles)

    # Early Stop
    early_stop = EarlyStop(early_stop_patience, min_steps=decay_episodes)

    # Buffer Filling
    print("*** Prefilling Buffer ***")
    prefill_buffer(env, agent, rendering=rendering, history_length=hist_len)
    # Now after buffering, set epsilon to the desired initial value
    agent.epsilon = epsilon0

    # Training
    print("\n\n*** Training Agent ***")
    train_online(env, agent, num_episodes, epsilon_schedule, early_stop,
                 history_length=hist_len, max_timesteps=max_timesteps,
                 model_dir=model_dir, tensorboard_dir=tensorboard_dir, tensorboard_suffix=name_suffix,
                 rendering=rendering,
                 skip_frames=skip_frames)

    # Testing
    print("\n\n*** Testing Agent ***")
    test_model(model_dir, name_suffix, hist_len=hist_len, conv_layers=conv_layers, fc_layers=fc_layers, n_test_episodes=15, verbose=False)
