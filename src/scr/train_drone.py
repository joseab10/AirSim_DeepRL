import sys
from os import path, makedirs

sys.path.append("../")
SCR_PATH = path.dirname(__file__)
sys.path.append(path.join(SCR_PATH,'..', 'lib'))
sys.path.append(path.join(SCR_PATH,'..', 'cls', 'airsim'))
sys.path.append(path.join(SCR_PATH,'..', 'cls', 'deep_rl'))
sys.path.append(path.join(SCR_PATH,'..', 'scr'))

import numpy as np
import tensorflow as tf

from cls.airsim.as_env import *
from cls.deep_rl.drl_dqn_agent import DRL_DQNAgent
from cls.deep_rl.drl_model import DRL_Model, DRL_TargetModel
from cls.deep_rl.drl_tb_eval import DRL_TB_Evaluation
from cls.deep_rl.drl_episode_stats import DRL_EpisodeStats
from cls.deep_rl.drl_schedule import *
from cls.deep_rl.drl_early_stop import DRL_EarlyStop

from datetime import datetime
import json




def run_episode(env, agent, deterministic,  do_training=True, do_prefill=False,
                max_timesteps=1000, verbose=False):

    stats = DRL_EpisodeStats()

    step = 1

    min_x = -4
    max_x = 20
    min_y = -4
    max_y = 4
    min_z = -8
    max_z = 0.1

    x_ini = np.random.uniform(min_x, max_x)
    y_ini = np.random.uniform(min_y, max_y)
    z_ini = np.random.uniform(min_z, max_z)

    state = env.reset(starting_position=(x_ini, y_ini, z_ini))
    state = state_preprocessing(state)

    terminal = False
    
    while not terminal:

        action_id = agent.act(state, deterministic)
        action = env.actid2str(action_id)

        if verbose:
            print('\tStep ', '{:7d}'.format(step), ' Action: ', action)


        next_state, reward, terminal = env.step(action_id)
        next_state = state_preprocessing(next_state)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        if do_prefill:
            agent._replay_buffer.add_transition(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or step > max_timesteps :
            break

        #if step % 100 == 0 and False:
            print('\t\tStep ', '{:4d}'.format(step), ' Reward: ', '{:4.4f}'.format(stats.episode_reward))

        step += 1

    return stats


def prefill_buffer(env, agent, max_timesteps = 1000, verbose=False):

    episode = 0
    while not agent._replay_buffer.has_min_items():

        episode += 1
        stats = run_episode(env, agent, deterministic=False, do_training=False, do_prefill=True,
                            max_timesteps=max_timesteps, verbose=verbose)

        print("Prefill Episode: ", '{:7d}'.format(episode), ' Reward: ', '{:4.0f}'.format(stats.episode_reward),
              ' Buffer Filled: ', '{:8d}'.format(agent._replay_buffer.len()))


def train_online(env, agent, num_episodes, epsilon_schedule, early_stop,
                 max_timesteps=1000,
                 ckpt_dir="./models_carracing", tensorboard_dir="./tensorboard",
                 verbose=False):
   
    if not path.exists(ckpt_dir):
        makedirs(ckpt_dir)

    tb_variables = [
                    {'plt_name': 'episode_reward'        , 'name': 'episode_reward'      , 'type': 'float32'},
                    {'plt_name': 'validation_reward'     , 'name': 'validation_reward'   , 'type': 'float32'},
                    {'plt_name': 'episode_reward_100'    , 'name': 'episode_reward_100'  , 'type': 'float32'},
                    {'plt_name': 'validation_reward_10'  , 'name': 'validation_reward_10', 'type': 'float32'},
                    {'plt_name': 'episode_duration'      , 'name': 'episode_duration'    , 'type': 'float32'},
                    {'plt_name': 'epsilon'               , 'name': 'epsilon'             , 'type': 'float32'}
                    ]

    for action in env.possible_actions:
        action_str = action.name
        tmp_action_var = {'plt_name': action_str, 'name': action_str, 'type': 'float32'}
        tb_variables.append(tmp_action_var)

    tb_dict = {'summaries': {
        'scalar': tb_variables
        }}

    tensorboard = DRL_TB_Evaluation(tensorboard_dir, session, tb_dict)

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
        #do_rendering = rendering

        # Validation (Deterministic)
        if i % 10 == 0:
            deterministic = True
            #if i % 100 == 0:
            #    do_rendering = True

        if epsilon_schedule is not None:
            agent.epsilon = epsilon_schedule(i)
       
        stats = run_episode(env, agent, max_timesteps=max_timesteps,
                            deterministic=deterministic, do_training=training, verbose=verbose)

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

        eval_dict = { "episode_reward"      : stats.episode_reward,
                      "validation_reward"   : valid_reward,
                      "episode_reward_100"  : train_reward_100,
                      "validation_reward_10": valid_reward_10,
                      "episode_duration"    : stats.episode_steps,
                      "epsilon"             : agent.epsilon
                      }

        for action in env.possible_actions:
            action_str = action.name
            action_id  = env.act2id(action)
            action_val = stats.get_action_usage(action_id)

            eval_dict[action_str] = action_val

        tensorboard.write_episode_data(i, eval_dict=eval_dict)

        # Save model every 10 steps
        if i % 10 == 0:
            agent.save(path.join(ckpt_dir, 'model.ckpt'))
            agent.save(path.join(ckpt_dir, 'model_%s' % datetime.now().strftime("%Y%m%d_%H%M%S") + '.ckpt'))

            # Increment max_steps
            env.max_steps += 100

        # Save model on the last step
        if i >= num_episodes - 1:
            agent.save(path.join(ckpt_dir, 'model.ckpt'))

        print('Episode ', ep_type, ': ', '{:7d}'.format(i + 1), ' Reward: ', '{:4.4f}'.format(stats.episode_reward))

        # Early Stopping
        early_stop.step(valid_reward_10)
        if early_stop.save_flag:
            if early_stop.stop:
                break
            else:
                agent.save(path.join(ckpt_dir, 'model.ckpt'))
                agent.save(path.join(ckpt_dir, 'model_earlystop_%s' % datetime.now().strftime("%Y%m%d_%H%M%S") + '.ckpt'))

    tensorboard.close_session()


def test_model(env, agent, model_ckpt_path, model_res_dir='res',
               max_timesteps=1000,
               n_test_episodes=15,
               verbose=True):

    agent.load(model_ckpt_path)

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, do_prefill=False, verbose=verbose,
                            max_timesteps=max_timesteps)
        episode_rewards.append(stats.episode_reward)
        print("Episode", " Reward:", stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()


    results_file = "results-%s" % datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    results_file = path.join(model_res_dir, results_file)

    if not path.exists(model_res_dir):
        makedirs(model_res_dir)

    with open(results_file, 'w') as fh:
        json.dump(results, fh)


def state_preprocessing(state):
    img_response = state['scene']
    pose = state['pose']

    img1d = np.fromstring(img_response.image_data_uint8, dtype=np.uint8)
    img1d = img1d.astype(np.float32)
    # reshape array to 4 channel image array H X W X 4
    img_rgba = img1d.reshape(img_response.height, img_response.width, 4)
    # original image is fliped vertically
    img_rgba = np.flipud(img_rgba)
    # get rid of alpha component
    camera_in = img_rgba[:, :, 0:3]
    camera_in = camera_in.reshape(1,img_response.height, img_response.width, 3)

    sensor_in = [pose.position.x_val,
                 pose.position.y_val,
                 pose.position.z_val,
                 pose.orientation.w_val,
                 pose.orientation.x_val,
                 pose.orientation.y_val,
                 pose.orientation.z_val]

    sensor_in = np.array(sensor_in)
    sensor_in = sensor_in.reshape((1,7))

    state = {'camera_in': camera_in,
             'sensor_in': sensor_in}

    return state




if __name__ == "__main__":

    import argparse

    cwd_dir  = path.dirname(__file__)
    proj_dir = path.join(cwd_dir, '..', '..')
    proj_dir = path.normpath(proj_dir)

    # Default Arguments
    model_dir = path.join(proj_dir, 'models')
    default_model_cfg_dir = path.join(model_dir, 'cfg')
    default_model_cpt_dir = path.join(model_dir, 'cpt')
    default_model_res_dir = path.join(model_dir, 'res')

    default_tb_dir        = path.join(proj_dir, 'tensorboard')

    default_model = 'knet1'
    default_model_ext = 'narq.json'

    default_env_dir = path.join(proj_dir, 'env')
    default_env_dir = path.normpath((default_env_dir))
    default_env = 'neighborhood'

    default_targets = 'default'

    # Argument Parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False, help='Train Model.')
    parser.add_argument('--test' , action='store_true', default=False, help='Test Model.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print action messages.')

    parser.add_argument('--df',      action='store', default=0.99, help='Past Rewards Discount Factor.', type=float)
    parser.add_argument('--bs',      action='store', default=100, help='Batch Size.',                   type=int)
    parser.add_argument('--episodes',action='store', default=10000, help='Maximum Number of Training Episodes.', type=int)

    # Model
    parser.add_argument('--cfg_dir', action='store', default=default_model_cfg_dir, help='Model Config Directory.')
    parser.add_argument('--cpt_dir', action='store', default=default_model_cpt_dir, help='Model Checkpoint Directory.')
    parser.add_argument('--res_dir', action='store', default=default_model_res_dir, help='Model Results Directory.')
    parser.add_argument('--tb_dir' , action='store', default=default_tb_dir       , help='TensorBoard Directory.')

    parser.add_argument('--model', action='store', default=default_model,
                        help='Model configuration name (without file extension).')

    parser.add_argument('--double_q', action='store_true', default=False,
                        help='Use Double Q-Learning.')
    parser.add_argument('--prefill_bs_pc', action='store', default=10, help='Prefill Batch Size Percentage.', type=float)

    # Environment
    parser.add_argument('--env_dir', action='store', default=default_env_dir,
                        help='AirSim Environment Directory.')
    parser.add_argument('--env', action='store', default=default_env,
                        help='AirSim Environment.')

    parser.add_argument('--step_sim', action='store_true', default=False,
                        help='Use Paused Simulation.')


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

    parser.add_argument('--quick_debug', action='store_true', default=False,
                        help='Run a smaller problem to test all functionalities.')


    # Targets
    parser.add_argument('--targets', action='store', default=default_targets, help='Target Positions.')
    




    args = parser.parse_args()

    train = args.train
    test = args.test

    if not (train or test):
        print('Either --train, --test or both must be selected')
        exit()

    model = args.model

    model_cfg_dir = args.cfg_dir
    model_cfg_dir = path.normpath(model_cfg_dir)
    model_cfg = path.join(model_cfg_dir, model + '.' + default_model_ext)
    model_cfg = path.normpath(model_cfg)

    model_ckpt_dir = path.join(args.cpt_dir, model)
    model_ckpt_dir = path.normpath(model_ckpt_dir)
    model_ckpt = path.join(model_ckpt_dir, 'model.ckpt')
    model_ckpt = path.normpath(model_ckpt)

    model_res_dir = path.join(args.res_dir, model)
    model_res_dir = path.normpath(model_res_dir)

    environment = Environments(args.env.upper())
    env_dir     = args.env_dir

    targets = args.targets

    # Render the carracing 2D environment
    rendering = args.render

    stepped_simulation = args.step_sim

    verbose = args.verbose
    quick_debug = args.quick_debug

    env = AS_Environment(env=environment, env_path=env_dir,rendering=rendering,stepped_simulation=stepped_simulation)
    #env.reset()
    print('\n\n***Environment Loaded and API connected***')

    # Targets
	tg_names = False
    
    if targets == 'mn_lines':
        targets = ['SM_PylonA_60M6', 'SM_PylonA_60M5', 'SM_PylonA_60M4', 'SM_PylonA_60M2', 'SM_Transformer2',
                   'SM_PylonA_60M2', 'SM_PylonA_60M3', 'SM_PylonA_60M7']
        tg_names = True
    elif targets == 'nh_lines':
        targets = []
    elif targets == 'nh_pools':
        targets = []
    else
        targets = [(20, 0, -5), (5, 5, -5), (10, -10, -8), (0, 0, -5)]

    if tg_names:
        env.add_targets_by_name(targets)
    else:
        env.add_targets_by_pos(targets)


    # Past Rewards Discount Factor (Gamma)
    discount_factor = args.df

    batch_size = args.bs
    if quick_debug:
        batch_size = 20

    Q = DRL_Model(model_file=model_cfg)
    Q_Target = DRL_TargetModel(model_file=model_cfg)

    num_actions = Q.num_outputs()

    double_q_learning = args.double_q

    buffer_capacity = 1000000
    if quick_debug:
        buffer_capacity = 10000

    prefill_bs_pc = args.prefill_bs_pc

    # Random Action Probability Distribution
    act_probabilities = np.zeros(num_actions)

    act_probabilities[DroneAct.VxBW.value] = 2
    act_probabilities[DroneAct.VxFW.value] = 10
    act_probabilities[DroneAct.VyBW.value] = 2
    act_probabilities[DroneAct.VyFW.value] = 10
    act_probabilities[DroneAct.VzBW.value] = 2
    act_probabilities[DroneAct.VzFW.value] = 30
    act_probabilities[DroneAct.YawL.value] = 10
    act_probabilities[DroneAct.YawR.value] = 10

    act_probabilities /= np.sum(act_probabilities)

    session = tf.Session()

    # Start with epsilon=1 for the buffering, so that all actions are random and in the specified probabilities
    # instead of randomly depending on the initialized parameters
    agent = DRL_DQNAgent(Q, Q_Target, num_actions, session=session,
                         discount_factor=discount_factor, batch_size=batch_size,
                         epsilon=1, act_probabilities=act_probabilities,
                         double_q_learning=double_q_learning,
                         replay_buffer_capacity=buffer_capacity, prefill_bs_percentage=prefill_bs_pc)

    print('\n***Agent instantiated***\n')


    # Perform Training
    if train:

        # Random Exploration Rate (Epsilon)
        epsilon0         = args.e_0
        min_epsilon      = args.e_min
        decay_function   = DRL_ScheduleFunctions(args.e_df.upper())
        cosine_annealing = args.e_ann
        annealing_cycles = args.e_acyc
        decay_episodes   = args.e_steps

        # Tensorboard
        tb_dir = path.join(args.tb_dir, model)
        tb_dir = path.normpath(tb_dir)

        # Early Stop
        early_stop_patience = args.patience

        # Maximum number of training Episodes
        num_episodes = args.episodes
        if quick_debug:
            num_episodes = 5

        max_timesteps = 10000
        if quick_debug:
            max_timesteps = 50

        # Exploration-vs-Exploitation Parameter (Epsilon) Schedule
        epsilon_schedule = DRL_Schedule(epsilon0, min_epsilon, decay_episodes, schedule_function=decay_function,
                                    cosine_annealing=cosine_annealing, annealing_cycles=annealing_cycles)

        # Early Stop
        early_stop = DRL_EarlyStop(early_stop_patience, min_steps=decay_episodes)

        # Buffer Filling
        print("\n*** Prefilling Buffer ***\n")
        prefill_buffer(env, agent, max_timesteps=prefill_bs_pc * batch_size, verbose=verbose)
        # Now after buffering, set epsilon to the desired initial value
        agent.epsilon = epsilon0

        # Training
        print("\n*** Training Agent ***\n")
        train_online(env, agent, num_episodes, epsilon_schedule, early_stop,
                     max_timesteps=max_timesteps,
                     ckpt_dir=model_ckpt_dir, tensorboard_dir=tb_dir, verbose=verbose)


    # Perform Testing
    if test:
        print("\n*** Testing Agent ***\n")

        n_test_episodes = 15
        if quick_debug:
            n_test_episodes = 3

        test_model(env, agent, model_ckpt, model_res_dir, n_test_episodes=n_test_episodes, max_timesteps=max_timesteps,
                   verbose=verbose)


    session.close()
    env.close()
