#! /Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5

from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np


def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.2

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--collect_data", action="store_true", default=False          , help="Collect the data in a pickle file.")
    parser.add_argument("--data_file"   , action="store"     , default='data.pkl.gzip', help="Filename of pickle file.")

    args = parser.parse_args()

    samples = {
        "state": [],
        "next_state": [],
    #    "reward": [],
        "action": [],
    #    "terminal" : [],
    }

    env = gym.make('CarRacing-v0').unwrapped

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    a = np.array([0.0, 0.0, 0.0]).astype('float32')
    
    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state = env.reset()
        while True:

            next_state, r, done, info = env.step(a)
            episode_reward += r

            # Added the preprocessing here
            samples["state"].append(rgb2gray(state.astype('float32')).astype('uint8'))            # state has shape (96, 96, 3)
            samples["action"].append(np.array(one_hot(np.array([action_to_id(a)]))).astype('uint8')[0])     # action has shape (1, 3)
            #samples["next_state"].append(rgb2gray(next_state.astype('float32')).astype('uint8'))
            #samples["reward"].append(r)
            #samples["terminal"].append(done)
            
            state = next_state
            steps += 1

            if steps % 1000 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            if args.collect_data and steps % 5000 == 0:
                print('... saving data')
                store_data(samples, "./data", file_name=args.file_name)
                save_results(episode_rewards, "./results")

                # <JAB>
                # Instead of storing everything in memory, reset and reload from the saved file
                # That way we can save Data in more than one session.
                samples = {
                    "state": [],
                #    "next_state": [],
                #    "reward": [],
                    "action": [],
                #    "terminal": [],
                }
                # </JAB>

            env.render()
            if done: 
                break
        
        episode_rewards.append(episode_reward)

    env.close()

    

   
