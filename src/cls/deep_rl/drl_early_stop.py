
class DRL_EarlyStop:
    def __init__(self, patience:int, stalled_derivative:float=0.1, min_steps:int=200):

        self.patience = patience
        self._min_steps = min_steps

        self.stop      = False
        self.save_flag = False

        self.steps = 0

        self._stalled_episodes = 0
        self._stalled_derivative = stalled_derivative

        self._notice_threshold  = 0.5
        self._warning_threshold = 0.75
        self._alarm_threshold   = 0.9

        self._J_t_1 = 0
        self._J_t_2 = 0

        self._dJ = 0

    def step(self, J):

        # Reward Derivative (Central Difference)
        self._dJ = (J - self._J_t_2) / 2

        self.steps +=1

        if self.steps > self._min_steps:
            if abs(self._dJ) < self._stalled_derivative:
                self._stalled_episodes += 1

            if self._stalled_episodes >= self.patience * self._alarm_threshold:
                print('\t\aEarly Stop ALARM!!: learning plateau detected for ', self._stalled_episodes, 'episodes.')
            elif self._stalled_episodes >= self.patience * self._warning_threshold:
                print('\tEarly Stop Warning: learning plateau detected for ', self._stalled_episodes, 'episodes.')
            elif self._stalled_episodes >= self.patience * self._notice_threshold:
                print('\tEarly Stop notice : learning plateau detected for ', self._stalled_episodes, 'episodes.')

        self._J_t_2 = self._J_t_1
        self._J_t_1 = J

        self._stop()

    def _stop(self):

        stop = False
        warning = False

        if self.steps > self._min_steps:
            if self._stalled_episodes >= self.patience:
                warning = True

                if self._dJ < - 5 * self._stalled_derivative:
                    print('EARLY STOP: FINISHING TRAINING! Decreasing reward after ',
                          self._stalled_episodes, ' stalled episodes.')
                    stop = True

                elif self._dJ > 5 * self._stalled_derivative:
                    self.reset(msg=True)

            elif abs(self._dJ) > 5 * self._stalled_derivative:
                self.reset()

        elif abs(self._dJ) > self._stalled_derivative:
            self.reset()

        self.stop      = stop
        self.save_flag = warning

    def reset(self, msg=False):
        self._stalled_episodes = 0
        if msg:
            print('\tEarly Stop notice : Stalled episodes reset back to 0.')
