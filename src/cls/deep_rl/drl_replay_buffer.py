from collections import namedtuple, deque
import numpy as np
import os
import gzip
import pickle

class DRL_ReplayBuffer:

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity=100000, min_fill=1000):
        self._data = namedtuple("DataBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=deque(maxlen=capacity),
                                actions=deque(maxlen=capacity),
                                next_states=deque(maxlen=capacity),
                                rewards=deque(maxlen=capacity),
                                dones=deque(maxlen=capacity))
        self._min_fill = min_fill


    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)


    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)

        batch_states      = np.array([self._data.states[i] for i in batch_indices])
        batch_actions     = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards     = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones       = np.array([self._data.dones[i] for i in batch_indices])

        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones


    def len(self):
        return len(self._data.dones)


    def has_min_items(self):
        if self.len() > self._min_fill:
            return True
        else:
            return False

    def dump(self, dir:str, file_name:str, append:bool=False):

        if not os.path.exists(dir):
            os.makedirs(dir)
        data_file = os.path.join(dir, file_name)

        if append:
            tmp_data = self._read_data(dir, file_name)

            tmp_data['states']      += self._data.states
            tmp_data['actions']     += self._data.actions
            tmp_data['next_states'] += self._data.next_states
            tmp_data['rewards']     += self._data.rewards
            tmp_data['dones']       += self._data.dones

            self._init_from_data(tmp_data)

        dump_data = {}
        dump_data['states']  = self._data.states
        dump_data['actions'] = self._data.actions
        dump_data['next_states'] = self._data.next_states
        dump_data['rewards'] = self._data.rewards
        dump_data['dones'] = self._data.dones


        with gzip.open(data_file, 'wb') as f:
            pickle.dump(dump_data, f)


    def load(self, dir:str, file_name:str):

        tmp_data = self._read_data(dir, file_name)

        self._init_from_data(tmp_data)



    def _read_data(self, dir:str, file_name:str):

        data_file = os.path.join(dir, file_name)

        if os.path.exists(data_file):
            with gzip.open(data_file, 'rb') as f:
                data = pickle.load(f)
                return data

        else:
            raise FileNotFoundError(data_file)

    def _init_from_data(self, data):

        self._data.states.clear()
        self._data.actions.clear()
        self._data.next_states.clear()
        self._data.rewards.clear()
        self._data.dones.clear()

        self._data.states.extend(data['states'])
        self._data.actions.extend(data['actions'])
        self._data.next_states.extend(data['next_states'])
        self._data.rewards.extend(data['rewards'])
        self._data.dones.extend(data['dones'])


# Class Tests
if __name__ == '__main__':

    def add_random_data(buffer, samples):

        state_shape = (28, 28, 3)
        num_actions = 5

        tmp_state = np.zeros(state_shape)

        tmp_done_states = [True, False]
        tmp_done_probabilities = [0.1, 0.9]


        for _ in range(samples):

            tmp_next_state = np.random.normal(0, 1, state_shape)
            tmp_reward = np.random.normal(0, 500, 1)
            tmp_done = np.random.choice(tmp_done_states, 1, p=tmp_done_probabilities)

            tmp_action_index = np.argmax(np.random.normal(0.5, 0.15, num_actions))
            tmp_action = np.zeros(num_actions)
            tmp_action[tmp_action_index] = 1

            buffer.add_transition(tmp_state, tmp_action, tmp_next_state, tmp_reward, tmp_done)

            tmp_state = tmp_next_state


    data_dir = os.path.join('..', '..', '..', 'data')
    data_file = 'test_buffer.pkl.gz'

    data = DRL_ReplayBuffer()
    add_random_data(data, 100)
    data.dump(data_dir, data_file)

    print('New Data Length: ', data.len())

    appended_data = DRL_ReplayBuffer()
    add_random_data(appended_data, 50)
    print('Appended Data Length: ', appended_data.len())
    appended_data.dump(data_dir, data_file, append=True)
    print('Appended Data Length: ', appended_data.len())

    loaded_data = DRL_ReplayBuffer()
    loaded_data.load(data_dir, data_file)
    print('Loaded Data Length: ', appended_data.len())

    batch_size = 64
    batch = loaded_data.next_batch(batch_size)
    print('Batch Data Length: ', len(batch[4]))


