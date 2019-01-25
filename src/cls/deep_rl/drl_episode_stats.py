import numpy as np

class DRL_EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []
        self.episode_steps = 0

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)
        self.episode_steps += 1

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))