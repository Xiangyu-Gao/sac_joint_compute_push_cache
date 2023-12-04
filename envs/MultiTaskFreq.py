import inspect
import os
import sys
import math
import random
import numpy as np
from gym import spaces

# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import chip_config, system_config
from tool.data_loader import load_data

tau = chip_config['tau']
fD = chip_config['fD']
u = chip_config['u']
num_freq = system_config['FQ']
num_task = system_config['F']
Cache = chip_config['C']

MAX_STEPS = 2000


class MultiTaskFreq(object):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            init_sys_state,  # initiated system state, S^I, S^O for each task, plus A(0) in range [0, n_task-1]
            task_set,  # a list of n_actions sublist, for each sublist it is [I, O, w]
            requests,   # the request task samples
            exp_case='case5',   # the experiment configuration
    ):
        super(MultiTaskFreq, self).__init__()
        self.sys_state = init_sys_state
        self.init_sys_state = init_sys_state
        self.task_set = task_set
        self.current_step = 0   # for restart the environment for bad cases
        self.global_step = 0    # for select the request sample
        self.requests = requests
        self._max_episode_steps = MAX_STEPS
        self.reactive_only = False

        # action: [c, b_trans, b_comp, dSO_f(all tasks)]
        # sys_state: [S_I(f) (all tasks), S_O(f) (all tasks), At], where At = [0, F-1]
        print("The Chosen eExperiment Configuration is: {}".format(exp_case))
        if exp_case == 'case5': # case 5: proactive transmit + compute, dynamic fD
            action_low = np.asarray([0, 0, 0] + [-1] * num_task)
        elif exp_case == 'case4':   # case 4: proactive transmit + compute, highest fD
            action_low = np.asarray([num_freq, 0, 0] + [-1] * num_task)
        elif exp_case == 'case3':   # case 3: proactive transmit only, highest fD
            action_low = np.asarray([num_freq, 0, num_task] + [0] * num_task)
        elif exp_case == 'case2':  # case 2: no cache, reactive only, dynamic fD
            action_low = np.asarray([0, num_task, num_task] + [0] * num_task)
            self.reactive_only = True
        elif exp_case == 'case1':  # case 1: no cache, reactive only, highest fD
            action_low = np.asarray([num_freq, num_task, num_task] + [0] * num_task)
            self.reactive_only = True
        action_high = np.asarray([num_freq, num_task, num_task] + [0] * num_task)
        observe_low = np.asarray([0] * (2 * num_task) + [0])
        observe_high = np.asarray([1] * (2 * num_task) + [num_task-1])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float16)
        self.observation_space = spaces.Box(low=observe_low, high=observe_high, dtype=np.float16)

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self.current_step += 1
        self.global_step += 1
        corrected_action, next_state, num_error, is_valid = self.correct_action(action)
        print(corrected_action, action, is_valid, self.sys_state, self.global_step)
        if is_valid:
            # calculate the observation based on action
            observation_, observe_details = self.calc_observation(corrected_action)
            reward = max(0, 10 - observation_ ** 2 / 1e11)
            if num_error > 0:
                reward -= 5
        else:
            self.global_step -= 1
            reward = -100

        # if not self.reactive_only:
        #     reward -= num_error * 1000
        #     if num_error == 0 and corrected_action[1] < num_task and corrected_action[2] < num_task:
        #         reward += 10000

        done = self.current_step > MAX_STEPS
        ob = np.array(next_state[0] + next_state[1] + [self.requests[int(self.global_step % len(self.requests))]])
        self.sys_state = ob

        return ob, reward, done, {'observe_detail': observe_details}

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.current_step = 0
        self.global_step = self.global_step * random.uniform(0, 1)
        self.sys_state = self.init_sys_state

        return self.sys_state

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""
        print(f'Step: {self.current_step}')

    def calc_observation(self, action):
        """
        calculate B_R + B_P + E_R + E_T
        """
        A_t = int(self.sys_state[-1])
        I_At = self.task_set[A_t][0]
        w_At = self.task_set[A_t][2]
        S_I_At = self.sys_state[A_t]
        S_O_At = self.sys_state[num_task + A_t]
        c_t = action[0]
        b_trans = action[1]
        b_comp = action[2]
        B_R = (1 - S_I_At) * (1 - S_O_At) * I_At / (tau - I_At * w_At / (c_t * fD))
        E_R = (1 - S_O_At) * u * (c_t * fD) ** 2 * I_At * w_At

        if self.reactive_only:
            return B_R + E_R, [B_R, E_R]

        E_P = 0
        B_P = 0
        for idx in range(num_task):
            if idx == b_comp:
                E_P += u * (c_t * fD) ** 2 * self.task_set[idx][0] * self.task_set[idx][2]
            if idx == b_trans:
                B_P += self.sys_state[idx] * self.task_set[idx][0] / tau

        return B_R + E_R + E_P + B_P, [B_R+B_P, E_R+E_P]

    def correct_action(self, action):
        """
        Input:
            action: [c, b_trans, b_comp, dSO_f(all tasks)]
            sys_state: [S_I(f) (all tasks), S_O(f) (all tasks), At], where At = [0, F-1]
        Constraints:
            1) I(f) * w(f) / tau <= M * fD     # basic system constraint (not check here)
            2) I(At) * w(At) / tau <= c(t)* fD,     when S_O(At)=0
            3) I(f) * w(f) / tau <= c(t) * fD,    when p_comp(f) > 0
            4) p_comp(f) <= S_O(At)
            5) p_comp(f) <= S_I(f)
            6) sum of  p_comp(f) <= 1 # for all f in [0, F-1] (naturally forced)
            7) p_trans(f) > S_I(f),      when p_trans(f) > 0
            8) sum of  p_trans(f) <= 1 # for all f in [0, F-1] (naturally forced)
            9) S_I(f, t+1) = max(0, min(1, S_I(f, t) - p_comp(f) + p_trans(f)))
            10) S_O(f, t+1) = max(0, min(1, S_O(f, t) + p_comp(f) + dS_O(f)))
            11) sum of I(f) * S_I(f, t+1) + O(f) * S_O(f, t+1) <= C (do nothing if obeys)
        """
        action = np.around(action)
        A_t = int(self.sys_state[-1])
        S_O_At = self.sys_state[num_task + A_t]
        I_At = self.task_set[A_t][0]
        w_At = self.task_set[A_t][2]
        c_t = action[0]
        b_trans = action[1]
        b_comp = action[2]
        num_error = 0
        is_valid = True
        S_I_next_all = []
        S_O_next_all = []
        S_I_cur_all = []
        S_O_cur_all = []

        sum_cache = 0

        # Constraint (2)
        if I_At * w_At / tau > (c_t * fD) and S_O_At == 0:
            c_t = min(num_freq, math.ceil(I_At * w_At / tau / fD))
            num_error += 1

        if self.reactive_only:
            corrected_action = [c_t, num_task, num_task] + [0] * num_task  # just for redundancy
            next_state = [list(self.sys_state[:num_task]), list(self.sys_state[num_task:2 * num_task])]
            return corrected_action, next_state, num_error, is_valid

        # Constraint (4)
        if S_O_At == 0 and b_comp != num_task:
            b_comp = num_task  # set p_comp for all tasks be 0
            num_error += 1

        for idx in range(num_task):
            dS_O = action[3 + idx]
            S_I = self.sys_state[idx]
            S_O = self.sys_state[num_task + idx]
            I = self.task_set[idx][0]
            O = self.task_set[idx][1]
            w = self.task_set[idx][2]

            # Constraint (3)
            if I * w / tau > c_t * fD and b_comp == idx:
                b_comp = num_task   # set p_comp for all tasks be 0
                num_error += 1

            # Constraint (5)
            if S_I == 0 and b_comp == idx:
                b_comp = num_task  # set p_comp for all tasks be 0
                num_error += 1

            # Constraint (7)
            if b_trans == idx and S_I != 0:
                b_trans = num_task  # set p_comp for all tasks be 0
                num_error += 1

            # Constraint (9), Constraint (10)
            S_I_next = max(0, min(1, S_I - int(b_comp == idx) + int(b_trans == idx)))
            S_O_next = max(0, min(1, S_O + int(b_comp == idx) + dS_O))
            S_I_next_all.append(S_I_next)
            S_O_next_all.append(S_O_next)
            S_I_cur_all.append(S_I)
            S_O_cur_all.append(S_O)

            # Constraint (11)
            sum_cache += (I * S_I_next + O * S_O_next)

        # Constraint (11)
        if sum_cache > Cache:
            is_valid = False

        # corrected action
        corrected_action = [c_t, b_trans, b_comp] + [0] * num_task    # just for redundancy

        # next state, if invalid action, keep next state same as current state
        if is_valid:
            next_state = [S_I_next_all, S_O_next_all]
        else:
            next_state = [S_I_cur_all, S_O_cur_all]

        return corrected_action, next_state, num_error, is_valid


if __name__ == '__main__':
    task_utils = load_data('./data/task4_utils.csv')
    task_set_ = task_utils.tolist()
    # for testing this script only
    env = MultiTaskFreq(
        init_sys_state=[0, 0, 0, 0, 0, 0, 0, 0, 1],
        task_set=task_set_,
        requests=[0, 1, 2, 3, 2, 3]
    )

    state_ = env.reset()
    action_ = [0] * 7
    obs, reward_, done, _ = env.step(action_)
    print(obs, reward_)
    env.render()
