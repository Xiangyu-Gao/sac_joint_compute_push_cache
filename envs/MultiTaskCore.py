import inspect
import os
import sys
import math
import random
import numpy as np
from random import seed
from gym import spaces

# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import chip_config, system_config, best_fDs
from tool.data_loader import load_data

tau = chip_config['tau']
fD = chip_config['fD']
u = chip_config['u']
num_core = system_config['M']
num_task = system_config['F']
Cache = chip_config['C']
weight = system_config['weight']

MAX_STEPS = 2000


class MultiTaskCore(object):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            init_sys_state,  # initiated system state, S^I, S^O for each task, plus A(0) in range [0, n_task-1]
            task_set,  # a list of n_actions sublist, for each sublist it is [I, O, w]
            requests,  # the request task samples
            channel_snrs,   # a list of snr value of the channel
            exp_case='case3',  # the experiment configuration, default the solution with proactive transmission
    ):
        super(MultiTaskCore, self).__init__()
        self.task_set = task_set
        self.channel_snrs = channel_snrs
        self.current_step = 0
        self.global_step = 0  # for select the request sample
        self.requests = requests
        self.sys_state = init_sys_state
        self.sys_state[-1] = self.requests[self.global_step % len(self.requests)]
        self.init_sys_state = init_sys_state
        self._max_episode_steps = MAX_STEPS
        self.popularity = [0] * num_task    # for heuristic solution
        self.last_use = [0] * num_task    # for heuristic solution
        self.reactive_only = False
        self.no_cache = False
        self.heuristic = False
        self.best_fDs = None
        self.exp_case = exp_case

        # action: [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
        # sys_state: [S_I(f) (all tasks), S_O(f) (all tasks), At], where At = [0, F-1]
        print("The Chosen eExperiment Configuration is: {}".format(exp_case))
        if exp_case == 'case1':  # case 1: no cache, reactive only, best fD choice (as baseline)
            self.reactive_only = True
            self.no_cache = True
            self.best_fDs = best_fDs
        elif exp_case == 'case2':  # case 2: no cache, reactive only, dynamic fD
            self.reactive_only = True
            self.no_cache = True
        elif exp_case == 'case4':  # case 4: with cache, reactive only, dynamic fD
            self.reactive_only = True
        elif exp_case == 'case6' or exp_case == 'case7':   # case 6, 7: MRU cache + LRU replace, MFU cache + LFU replace
            self.heuristic = True

        self.sample_low = np.asarray([-1] * (3 * num_task + 1))
        self.sample_high = np.asarray([1] * (3 * num_task + 1))
        self.observe_low = np.asarray([0] * (2 * num_task) + [0])
        self.observe_high = np.asarray([1] * (2 * num_task) + [num_task - 1])
        self.action_space = spaces.Box(low=self.sample_low, high=self.sample_high, dtype=np.float16)
        self.observation_space = spaces.Box(low=self.observe_low, high=self.observe_high, dtype=np.float16)

        # [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
        if self.exp_case == 'case1':  # case 1: no cache, reactive only, best fD choice (as baseline)
            self.action_low = np.asarray([0] * (3 * num_task + 1))
            self.action_high = np.asarray([num_core] + [0] * (3 * num_task))
        elif self.exp_case == 'case2':  # case 2: no cache, reactive only, dynamic fD
            self.action_low = np.asarray([0] * (3 * num_task + 1))
            self.action_high = np.asarray([num_core] + [0] * (3 * num_task))
        elif self.exp_case == 'case3':  # case 3: proactive transmit, dynamic fD
            self.action_low = np.asarray([0] * (num_task + 1) + [-1] * (2 * num_task))
            self.action_high = np.asarray([num_core] + [1] * (3 * num_task))
        elif self.exp_case == 'case4':  # case 4: with cache, reactive only, dynamic fD
            self.action_low = np.asarray([0] * (num_task + 1) + [-1] * (2 * num_task))
            self.action_high = np.asarray([num_core] + [0] * num_task + [1] * (2 * num_task))
        elif self.exp_case == 'case6':  # case 6: with cache, most recently used cache, least recently used replace,
            # fixed computing cores
            self.action_low = np.asarray([int(num_core * 3 / 4)] + [0] * (3 * num_task))
            self.action_high = np.asarray([int(num_core * 3 / 4)] + [0] * (3 * num_task))
        elif self.exp_case == 'case7':  # case 7: with cache, most frequently used cache, least frequently used replace,
            # fixed computing cores
            self.action_low = np.asarray([int(num_core * 3 / 4)] + [0] * (3 * num_task))
            self.action_high = np.asarray([int(num_core * 3 / 4)] + [0] * (3 * num_task))

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
        done = False
        self.last_use[int(self.sys_state[-1])] = (self.current_step - 1)
        self.popularity[int(self.sys_state[-1])] += 1

        action, prob_action = self.sample2action(action)

        valid, action = self.check_action_validity(action, prob_action)
        # print(action, self.sys_state, self.last_use, valid)

        # calculate the observation based on action
        observation_, observe_details, details2 = self.calc_observation(action)
        if self.current_step > MAX_STEPS:
            done = True

        obs = self.next_state(action, valid)
        self.sys_state = obs    # update the system state for the nex
        self.sys_state[-1] = self.requests[self.global_step % len(self.requests)]

        # reward_ = - observation_ ** 2 / 1e12
        reward_ = - observation_ / 1e6
        action = self.action2sample(action)

        return self.scale_state(self.sys_state), reward_, done, {'observe_detail': observe_details, 'action': action}

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.current_step = 0
        self.global_step -= 1
        self.sum_Comp = 0
        self.sum_Trans = 0
        self.popularity = [0] * num_task    # for heuristic solution
        self.last_use = [0] * num_task   # for heuristic solution
        self.sys_state = self.init_sys_state.copy()
        self.sys_state[-1] = self.requests[self.global_step % len(self.requests)]

        return self.scale_state(self.sys_state)

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""
        print(f'Step: {self.global_step}')

    def calc_observation(self, action):
        """
        # action: [CR_At, CP_f (all tasks), b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
        # object: calculate B_R + B_P + E_R + E_T
        """
        A_t = int(self.sys_state[-1])
        snr_t = self.channel_snrs[self.global_step % len(self.channel_snrs)]
        I_At = self.task_set[A_t][0]
        w_At = self.task_set[A_t][2]
        S_I_At = self.sys_state[A_t]
        S_O_At = self.sys_state[num_task + A_t]
        C_R_At = action[0]
        # print(action)
        if S_I_At == 1 or S_O_At == 1:
            B_R = 0
        else:
            B_R = (1 - S_I_At) * (1 - S_O_At) * I_At / ((tau - I_At * w_At / (C_R_At * fD)) * math.log2(1 + snr_t))
        E_R = (1 - S_O_At) * u * (C_R_At * fD) ** 2 * I_At * w_At

        if self.reactive_only:
            return B_R + E_R * weight, [B_R, E_R], [B_R, 0, E_R, 0]

        E_P = 0
        B_P = 0
        for idx in range(num_task):
            C_P = 0
            E_P += u * (C_P * fD) ** 2 * self.task_set[idx][0] * self.task_set[idx][2]  # always 0 since C_P is zero
            B_P += self.task_set[idx][0] * action[1 + idx] / (tau * math.log2(1 + snr_t))

        return B_R + B_P + (E_R + E_P) * weight, [B_R + B_P, E_R + E_P], [B_R, B_P, E_R, E_P]

    def sample2action(self, action):
        unit = [2.0 / (self.action_high[idx] - self.action_low[idx] + 1) for idx in range(len(self.action_high))]
        rescale_action = []
        prob_action = []
        for ide, elem in enumerate(action):
            rescale_action.append(min(self.action_low[ide] + (elem + 1) // unit[ide], self.action_high[ide]))
            prob_action.append((elem + 1) / 2)

        return rescale_action, prob_action

    def action2sample(self, action):
        unit = [2.0 / (self.action_high[idx] - self.action_low[idx] + 1) for idx in range(len(self.action_high))]
        sample = []
        for ide, elem in enumerate(action):
            sample.append((elem - self.action_low[ide] + 0.5) * unit[ide] - 1)

        return sample

    def scale_state(self, state):
        # Scale to [-1, 1]
        length = self.observe_high - self.observe_low
        scaled_state = []
        for idx, elem in enumerate(state):
            scaled_state.append(2 * (elem - self.observe_low[idx]) / length[idx] - 1)

        return scaled_state

    def next_state(self, action, is_valid=True):
        """
        action: [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
        calculate the system state for t+1, S_I(f), S_O(f); note that the A(t+1) is set as 0 which needs to be updated
        S_I(f, t+1) = S_I(f, t) + dS_I(f, t)
        S_O(f, t+1) = S_O(f, t) + dS_O(f, t)
        """
        if not is_valid:
            # Not do update when the action is not valid
            return self.sys_state

        next_state = [0] * len(self.sys_state)
        for idx in range(num_task):
            S_I = self.sys_state[idx]
            S_O = self.sys_state[num_task + idx]
            dS_I = action[1 + num_task + idx]
            dS_O = action[1 + num_task * 2 + idx]
            next_state[idx] = S_I + dS_I
            next_state[num_task + idx] = S_O + dS_O
            assert 0 <= (S_I + dS_I) <= 1 and 0 <= (S_O + dS_O) <= 1

        return np.array(next_state)

    def check_action_validity(self, action, prob_action):
        """
        Input:
            action: [CR_At, b_f (all tasks), dSI_f (all tasks), dSO_f(all tasks)]
            sys_state: [S_I(f) (all tasks), S_O(f) (all tasks), At], where At = [0, F-1]
        Constraints:
            1) I(f) * w(f) / tau <= M * fD     # system constraint (not check here)
            2) I(At) * w(At) / tau <= C_R(At) * fD,     when S_O(At)=0
            3) C_R(f) = 0,  for all f not At, or S_O(f)=1
            4) I(f) * w(f) / tau <= C_P(f) * fD,    when C_P(f) > 0
            5) C_P(f) <= S_I(f) * M
            6) sum of C_R(f) + C_P(f) <= M
            7) -S_I(f) <= dS_I(f) <= min{b(f), 1-S_I(f)}
            8) -S_O(f) <= dS_O(f) <= min{C_R(f)+C_P(f), 1-S_O(f)}
            9) sum of I(f) * (S_I(f) + dS_I(f)) + O(f) * (S_O(f) +dS_O(f)) <= C
        """
        A_t = int(self.sys_state[-1])
        S_O_At = self.sys_state[num_task + A_t]
        I_At = self.task_set[A_t][0]
        w_At = self.task_set[A_t][2]
        CR_At = action[0]

        b_f = action[1:1 + num_task].copy()
        dS_I_f = action[1 + num_task:1 + num_task * 2].copy()
        dS_O_f = action[1 + num_task * 2:1 + num_task * 3].copy()

        b_f_prob = prob_action[1:1 + num_task].copy()
        dS_I_f_prob = prob_action[1 + num_task:1 + num_task * 2].copy()
        dS_O_f_prob = prob_action[1 + num_task * 2:1 + num_task * 3].copy()

        S_I_f = self.sys_state[:num_task].copy()
        S_O_f = self.sys_state[num_task:num_task * 2].copy()
        I_f = [self.task_set[idx][0] for idx in range(num_task)]
        O_f = [self.task_set[idx][1] for idx in range(num_task)]
        # w_f = [self.task_set[idx][2] for idx in range(num_task)]

        b_f_new = [0] * num_task
        dS_I_f_new = [0] * num_task
        dS_O_f_new = [0] * num_task

        # Constraint (2)
        if I_At * w_At / tau > (CR_At * fD) and S_O_At == 0:
            CR_At = min(math.ceil(I_At * w_At / tau / fD), num_core)
        elif S_O_At == 1:
            CR_At = 0
        # choose best fD for reactive processing (only for case 1)
        if self.best_fDs is not None:
            CR_At = self.best_fDs[A_t]
        C_R_f = [0] * num_task
        C_R_f[A_t] = CR_At
        action[0] = int(CR_At)

        # for non-cache solutions
        if self.no_cache:
            return True, action

        # for heuristic solution
        if self.heuristic:
            if self.exp_case == 'case6':    # MRU cache + LRU replace
                # cache the input data of most recently used if it has not been cached
                most_interest_A = np.argmin(np.abs(np.asarray(self.last_use) - self.current_step))
                indic = np.where(S_I_f == 1)[0]
                least_interest_A = None
                if indic.size > 0:
                    least_interest_A = indic[np.argmax(np.abs(np.asarray(self.last_use)[indic] - self.current_step))]

            elif self.exp_case == 'case7':  # MFU cache + LFU replace
                # cache the input data of most frequently used if it has not been cached
                most_interest_A = np.argmax(np.asarray(self.popularity))
                indic = np.where(S_I_f == 1)[0]
                least_interest_A = None
                if indic.size > 0:
                    least_interest_A = indic[np.argmin(np.asarray(self.popularity)[indic])]

            if S_I_f[most_interest_A] == 0:
                dS_I_f_new[most_interest_A] = 1
                if most_interest_A != A_t:
                    b_f_new[most_interest_A] = 1  # proactive transmission
                is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
                # remove the cache for least frequently used task if it has cache
                if is_cache_exceed and least_interest_A is not None:
                    if S_I_f[least_interest_A] == 1:
                        dS_I_f_new[least_interest_A] = -1

            is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
            action[1:1 + num_task] = b_f_new.copy()
            action[1 + num_task:1 + num_task * 2] = dS_I_f_new.copy()

            return not is_cache_exceed, action

        # below are for non-heuristic, SAC solution
        bf_sort_idx = np.argsort(b_f_prob)[::-1]

        if b_f[bf_sort_idx[0]] > 0 and S_I_f[bf_sort_idx[0]] + S_O_f[bf_sort_idx[0]] < 1:
            b_f_new[bf_sort_idx[0]] = 1

        # give the action correction for non-reactive-only methods
        if not self.reactive_only:
            action[1:1 + num_task] = b_f_new.copy()

        push_idx = [idp for idp in range(num_task)] + [idp for idp in range(num_task)]
        push_IO_indc = [0] * num_task + [1] * num_task   # 0 for S_I and 1 for S_O
        push_prob = dS_I_f_prob + dS_O_f_prob

        for idx, b in enumerate(b_f_new):
            if b == 1:
                # when b = 1, dS_I only be 1, dS_O >= 0, best policy 3
                dS_I_f_new[idx] = 1
                # dS_O_f_new[idx] = 0

        # Constraint (9)
        is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)

        if is_cache_exceed:
            drop_sort = np.argsort(push_prob)
            for idx in list(drop_sort):
                if b_f_new[push_idx[idx]] > 0:
                    continue
                if push_IO_indc[idx] == 0 and S_I_f[push_idx[idx]] > 0:
                    dS_I_f_new[push_idx[idx]] = -1
                elif push_IO_indc[idx] == 1 and S_O_f[push_idx[idx]] > 0:
                    dS_O_f_new[push_idx[idx]] = -1
                is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
                if not is_cache_exceed:
                    break

        if not is_cache_exceed:
            push_sort = np.argsort(push_prob)[::-1]
            for idx in list(push_sort):
                if b_f_new[push_idx[idx]] > 0:
                    continue
                if push_IO_indc[idx] == 0 and S_I_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                    dS_I_f_new[push_idx[idx]] = 1
                elif push_IO_indc[idx] == 1 and S_O_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                    dS_O_f_new[push_idx[idx]] = 1
                is_cache_exceed = self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, dS_I_f_new, dS_O_f_new)
                if is_cache_exceed:
                    # convert it back for this change
                    if push_IO_indc[idx] == 0 and S_I_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                        dS_I_f_new[push_idx[idx]] = 0
                    elif push_IO_indc[idx] == 1 and S_O_f[push_idx[idx]] < 1 and C_R_f[push_idx[idx]] > 0:
                        dS_O_f_new[push_idx[idx]] = 0
                    break

        # give the action correction for non-reactive-only methods
        if not self.reactive_only:
            action[1 + num_task:1 + num_task * 2] = dS_I_f_new.copy()
            action[1 + num_task * 2:1 + num_task * 3] = dS_O_f_new.copy()
        else:
            for idx, b in enumerate(b_f):
                dS_I_f[idx] = max(-S_I_f[idx], min(dS_I_f[idx], min(C_R_f[idx] + b, 1 - S_I_f[idx])))
                dS_O_f[idx] = max(-S_O_f[idx], min(dS_O_f[idx], min(C_R_f[idx], 1 - S_O_f[idx])))
            action[1 + num_task:1 + num_task * 2] = dS_I_f.copy()
            action[1 + num_task * 2:1 + num_task * 3] = dS_O_f.copy()

        # print(action[1 + num_task:1 + num_task * 2], action[1 + num_task * 2:1 + num_task * 3], self.sys_state)

        if self.test_cache_exceed(I_f, O_f, S_I_f, S_O_f, action[1+num_task:1+num_task*2], action[1+num_task*2:1+num_task*3]):
            return False, action
        else:
            return True, action

    def test_cache_exceed(self, I_f, O_f, S_I_f, S_O_f, dS_I_f, dS_O_f):
        sum_cache = np.sum(np.asarray(I_f) * (np.asarray(S_I_f) + np.asarray(dS_I_f)) +
                           np.asarray(O_f) * (np.asarray(S_O_f) + np.asarray(dS_O_f)))
        if sum_cache > Cache:
            is_cache_exceed = True
        else:
            is_cache_exceed = False

        return is_cache_exceed


if __name__ == '__main__':
    task_utils = load_data('./data/task4_utils.csv')
    task_set_ = task_utils.tolist()
    channel_snrs = load_data('./data/one_snrs.csv')
    # for testing this script only
    env = MultiTaskCore(
        init_sys_state=[0, 0, 0, 0, 0, 0, 0, 0, 1],
        task_set=task_set_,
        requests=[0, 1, 2, 3, 2, 3],
        channel_snrs=channel_snrs
    )

    state_ = env.reset()
    action_ = [0] * 13
    obs, reward_, done, _ = env.step(action_)
    print(obs, reward_)
    env.render()
