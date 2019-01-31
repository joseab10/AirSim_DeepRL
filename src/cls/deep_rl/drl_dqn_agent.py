import tensorflow as tf
import numpy as np
from drl_replay_buffer import DRL_ReplayBuffer

class DRL_DQNAgent:

    def __init__(self, Q_model, Q_target_model, num_actions:int,
                 session,
                 # HyperParameters
                 discount_factor:float=0.99, batch_size:int=64, epsilon:float=0.05,
                 act_probabilities = None, double_q_learning = False,
                 replay_buffer_capacity:int=100000,
                 # Imitation Learning
                 combined_imitation_learning:bool=False, imitation_data_dir:str='',
                 imitation_data_file:str='', imitation_learning_prob:int=0,
                 # Buffer Pre-fill percentage (1=100% of batch size)
                 prefill_bs_percentage=5):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q_model: Action-Value function estimator (Neural Network)
            Q_target_model: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        self._Q_model = Q_model
        self._Q_target_model = Q_target_model
        
        self.epsilon = epsilon

        self._num_actions = num_actions
        self._batch_size = batch_size
        self._discount_factor = discount_factor

        # define replay buffer
        self._replay_buffer = DRL_ReplayBuffer(capacity=replay_buffer_capacity, min_fill=prefill_bs_percentage * batch_size)

        # Define Expert Data Buffer
        self._combined_imitation_data = combined_imitation_learning
        self._imitation_learning_prob = imitation_learning_prob
        if self._combined_imitation_data:
            self._expert_data = DRL_ReplayBuffer()
            self._expert_data.load(imitation_data_dir, imitation_data_file)

        # Start tensorflow session
        self._sess = session
        self._sess.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver(max_to_keep=50)

        if act_probabilities is None:
            self._act_probabilities = np.ones(num_actions) / num_actions
        else:
            self._act_probabilities = act_probabilities

        self._double_q_learning = double_q_learning


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        self._replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # Let the buffer fill up, otherwise we will burn up a lot of $#!+¥ states early on
        if self._replay_buffer.has_min_items():

            if self._combined_imitation_data and self._imitation_learning_prob > 0:
                expert_batch_size = int(self._batch_size * self._imitation_learning_prob)
                replay_batch_size = self._batch_size - expert_batch_size

                buffer = self._expert_data.next_batch(expert_batch_size)
                replay_buffer = self._replay_buffer.next_batch(replay_batch_size)

                for i in range(len(buffer)):
                    buffer[i].extend(replay_buffer[i])

            else:
                buffer = self._replay_buffer.next_batch(self._batch_size)

            batch_states      = buffer[0]
            batch_actions     = buffer[1]
            batch_next_states = buffer[2]
            batch_rewards     = buffer[3]
            batch_dones       = buffer[4]

            non_terminal_states = np.logical_not(batch_dones)

            if self._double_q_learning:
                a_predictions = self._Q_model.predict(self._sess, batch_next_states)
                a_predictions = np.argmax(a_predictions, axis=1)
                action_indexes = [np.arange(len(a_predictions)),a_predictions]
                q_predictions = self._Q_target_model.predict(self._sess, batch_next_states)
                q_predictions = q_predictions[action_indexes]

            else:
                q_predictions = self._Q_target_model.predict(self._sess, batch_next_states)
                q_predictions = np.max(q_predictions, axis=1)

            td_target = batch_rewards
            # If episode is not finished, add predicted Q values to the current rewards
            td_target[non_terminal_states] += self._discount_factor * q_predictions[non_terminal_states]

            # Update Step

            update_dict={
                'actions': batch_actions,
                'targets' : td_target
            }
            if isinstance(batch_states[0], dict):
                for key, _ in batch_states[0].items():
                    update_dict[key] = np.array(np.stack([batch_states[i][key][0] for i in range(self._batch_size)]))
            else:
                update_dict['state'] = batch_states

            self._Q_model.update(self._sess, update_dict)#batch_states, batch_actions, td_target)
            self._Q_target_model.update(self._sess)
   

    def act(self, state, deterministic:bool):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            action_id = np.argmax(self._Q_model.predict(self._sess, state))

        else:
            action_id = np.random.choice(np.arange(self._num_actions), p=self._act_probabilities)
          
        return action_id


    def load(self, file_name:str):
        self._saver.restore(self._sess, file_name)

    def save(self, file_name:str):
        self._saver.save(self._sess, file_name)