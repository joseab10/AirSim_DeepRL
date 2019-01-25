import numpy as np
from enum import Enum


class DRL_ScheduleFunctions(Enum):
    Constant    = 'CONSTANT'
    Linear      = 'LINEAR'
    Exponential = 'EXPONENTIAL'


class DRL_Schedule:

    def __init__(self, y_0, y_1, t, schedule_function:DRL_ScheduleFunctions=DRL_ScheduleFunctions.Exponential,
                 # Cosine Annealing Parameters
                 cosine_annealing=True, annealing_cycles=3,
                 # Stepping Parameters
                 steps=0):

        self.y_0 = y_0
        self.y_1 = y_1
        self.t   = t

        if schedule_function == DRL_ScheduleFunctions.Constant:
            self._decay_function = self._constant
            self.epsilon = self._constant
            self.y_0 = y_1

        elif schedule_function == DRL_ScheduleFunctions.Linear:
            self._decay_function = self._linear

        elif schedule_function == DRL_ScheduleFunctions.Exponential:
            self._decay_function = self._exponential
            # Due to the exponential aproaches asymptotically the final value,
            # this factor is to lower the asymptote a little bit to allow for an exact minimum value of epsilon after t epochs,
            # otherwise, it would just aproximate e_min, but never actually reach it
            self.exponential_k = 0.98
            sign = 1
            if y_1 > y_0:
                self.exponential_k = 2 - self.exponential_k
                sign = -1

            # Exponential multiplicative factor (e ^ (alpha * x)) to have a nice continuous function (almost tangent to e_min)
            # within the whole training interval while also reaching e_min at the end.
            self.exponential_factor = np.log(
                (sign * (1 - self.exponential_k) * y_1) / (self.exponential_k * y_1 + y_0)) / self.t

        else:
            raise NotImplemented("Error: Decay Function not implemented.")


        if steps > 0 and schedule_function != DRL_ScheduleFunctions.Constant:

            self.step_interval = self.t / steps
            tmp_x = np.arange(steps) * self.step_interval

            if schedule_function == DRL_ScheduleFunctions.Linear:
                self.y_1 = self._linear(self.t + self.step_interval)
                self.step_table = self._linear(tmp_x)

            if schedule_function == DRL_ScheduleFunctions.Exponential:
                self.y_1 = self._exponential(self.t + self.step_interval)
                self.step_table = self._exponential(tmp_x)

            self.y_1 = y_1

            self._decay_function = self._step


        # Cosine Annealing Parameters
        #----------------------------
        if cosine_annealing and schedule_function != DRL_ScheduleFunctions.Constant:
            self.epsilon = self._cos
        else:
            self.epsilon = self._decay_function

        # Number of full Peaks the cosine will make within the training interval
        self.annealing_cycles = annealing_cycles


    def _constant(self, x):
        if isinstance(x, np.ndarray):
            return self.y_1 * np.ones_like(x)

        else:
            return self.y_1

    def _linear(self, x):
        return (self.y_0 - ((self.y_0 - self.y_1) / self.t) * x)

    def _exponential(self, x):
        power = x * self.exponential_factor
        exp = (self.y_0 - (self.exponential_k * self.y_1)) * np.exp(power) + (self.exponential_k * self.y_1)

        if isinstance(exp, np.ndarray):
            if self.y_0 > self.y_1:
                exp[exp < self.y_1] = self.y_1
            else:
                exp[exp > self.y_1] = self.y_1
            decay = exp

        else:
            decay = exp
            if self.y_0 > self.y_1 and decay < self.y_1:
              decay = self.y_1

        return decay

    def _step(self, x):

        index = x // self.step_interval

        if isinstance(index, np.ndarray):
            index = index.astype(int)
            index[index < 0] = 0
            index[index >= len(self.step_table)] = len(self.step_table) - 1

            return np.array([self.step_table[i] for i in index])
        else:
            index = int(index)
            return self.step_table[index]

    def _cos(self, x):

        decay = self._decay_function(x)
        # Cosine Amplitude (Height)
        a = (decay - self.y_1) / 2

        # Cosine Function Centerline Offset (around which the cosine oscillates)
        b = (decay + self.y_1) / 2

        return (a * np.cos((2 * self.annealing_cycles + 1) * (np.pi / self.t) * x)) + b

    def _piecewise(self, x):

        y = self.epsilon(x)

        if isinstance(x, np.ndarray):
            indexes = np.where(x < 0)
            y[indexes] = self.y_0
            indexes = np.where(x > self.t)
            y[indexes] = self.y_1

        else:
            if x < 0:
                y = self.y_0
            elif x > self.t:
                y = self.y_1

        return y

    def __call__(self, x):
        return self._piecewise(x)


# Class Tests
if __name__ == '__main__':

    from matplotlib import pyplot as plt

    e_0 = 0.9
    e_min = 0.05
    epochs = 100

    annealing_cycles = 5

    x = np.arange(0, 200, 1)

    decay_functions = [
        {'decay': DRL_ScheduleFunctions.Constant   , 'annealing':  True, 'steps':  0, 'label': 'Constant'},
        {'decay': DRL_ScheduleFunctions.Linear     , 'annealing': False, 'steps':  0, 'label': 'Linear'},
        {'decay': DRL_ScheduleFunctions.Exponential, 'annealing': False, 'steps':  0, 'label': 'Exponential'},
        # Stepped
        {'decay': DRL_ScheduleFunctions.Linear     , 'annealing': False, 'steps': 10, 'label': 'Linear Step'},
        {'decay': DRL_ScheduleFunctions.Exponential, 'annealing': False, 'steps': 10, 'label': 'Exponential Step'},
        # Annealed
        {'decay': DRL_ScheduleFunctions.Linear     , 'annealing':  True, 'steps':  0, 'label': 'Linear Annealing'},
        {'decay': DRL_ScheduleFunctions.Exponential, 'annealing':  True, 'steps':  0, 'label': 'Exponential Annealing'},
        # Stepped and Annealed
        {'decay': DRL_ScheduleFunctions.Linear     , 'annealing':  True, 'steps': 10, 'label': 'Linear Step Annealing'},
        {'decay': DRL_ScheduleFunctions.Exponential, 'annealing':  True, 'steps': 10, 'label': 'Exponential Step Annealing'},
    ]

    fig = plt.figure(1)
    plot = fig.add_subplot(1, 1, 1)

    for function in decay_functions:
        epsilon = DRL_Schedule(e_0, e_min, epochs, schedule_function=function['decay'], cosine_annealing=function['annealing'],
                           steps=function['steps'], annealing_cycles=annealing_cycles)

        y = epsilon(x)

        plot.plot(x, y, label=function['label'])

    fig.legend()
    fig.show()
