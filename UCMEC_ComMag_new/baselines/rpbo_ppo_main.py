import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import math
from gym import spaces
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
import scipy.io as sio
import pandas as pd
from scipy.io import loadmat
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output

runtime_list = []

# Initialization
M = 10  # number of users
N = 10  # number of APs
P_max = 1.3  # maximum transmit power of user / pilot power
P_min = 0  # minimum transmit power of user
P_t = 1  # test transmit power of user
sinr_limit = 5000000

# locations are randomly sampling from 1000x1000 m^2 square area
# locations_users = np.random.random_sample([M, 2]) * 1000  # 2-D location of users
# locations_aps = np.random.random_sample([N, 2]) * 1000  # 2-D location of APs
loaded_locations_users = np.load('sinr_data/locations_users.npz', 'rb')
locations_users = (loaded_locations_users['arr_0'])
loaded_locations_aps = np.load('sinr_data/locations_aps.npz', 'rb')
locations_aps = (loaded_locations_aps['arr_0'])

# locations of single BS
# locations_BS = np.random.random_sample([1, 2]) * 1000
# loaded_locations_BS = np.load('sinr_data/locations_aps.npz', 'rb')
# locations_BS = (loaded_locations_BS['arr_0'])

# calculate distance between APs and users MxN matrix
distance_matrix = np.zeros([M, N])
# distance_matrix_single = np.zeros([1, M])
for i in range(M):
    for j in range(N):
        distance_matrix[i, j] = math.sqrt((locations_users[i, 0] - locations_aps[j, 0]) ** 2
                                          + (locations_users[i, 1] - locations_aps[j, 1]) ** 2)

# for i in range(M):
#     distance_matrix_single[0, i] = math.sqrt((locations_users[i, 0] - locations_BS[0, 0]) ** 2
#                                              + (locations_users[i, 1] - locations_BS[0, 1]) ** 2)

# edge computing parameter
# users parameter
# eta = 10e-27  # effective switched capacitance
# P_idle = 0.3
C_user = np.random.uniform(2e9, 5e9, [1, M])  # computing resource of users  in Hz

# edge server parameter
C_edge = np.random.uniform(10e9, 20e9)  # computing resource of edge server in Ghz


# C_edge = 3.7

def live_plot(data, plot_int):
    clear_output(wait=True)
    plt.plot(data)
    plt.xlabel(f'Episode batch (each batch consists of {plot_int} episodes)')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward per {plot_int} Episodes')
    plt.show()


def rate_calculate(m_, n_, p_):  # calculate SINR of all users under cell-free MIMO   -> 1xM vector
    # shadow loss calculation
    sigma_s = 8  # dB
    a_shadow = np.random.rand(1, n_)
    b_shadow = np.random.rand(1, m_)
    delta_shadow = random.random()  # random generate form [0,1]
    theta_shadow = np.zeros([m_, n_])
    for i in range(m_):
        for j in range(n_):
            theta_shadow[i, j] = math.sqrt(delta_shadow) * a_shadow[0, j] + math.sqrt(1 - delta_shadow) * b_shadow[
                0, i]  # MxN matrix

    # three slope path-loss calculation
    f_carrier = 1.9 * math.pow(10, 9)  # carrier frequency in Hz
    ah_ap = 15  # antenna height of AP
    ah_user = 1.65  # antenna height of user
    L = 46.3 + 33.9 * np.log10(f_carrier) - 13.82 * np.log10(ah_ap) - (
            1.11 * np.log10(f_carrier) - 0.7) * ah_user + 1.56 * np.log10(f_carrier) - 0.8
    d_0 = 10
    d_1 = 50
    pl = np.zeros([m_, n_])
    beta = np.zeros([m_, n_])  # large-scale fading
    w = 20e6  # system bandwidth in Hz
    for i in range(m_):
        for j in range(n_):
            if distance_matrix[i, j] > d_1:
                pl[i, j] = -L - 35 * np.log10(distance_matrix[i, j])
            elif d_0 <= distance_matrix[i, j] <= d_1:
                pl[i, j] = -L - 10 * np.log10(d_1 ** 1.5 * (distance_matrix[i, j]) ** 2)
            else:
                pl[i, j] = -L - 10 * np.log10(d_1 ** 1.5 * d_0 ** 2)

            # beta[i, j] = 10 ** (pl[i, j] / 10) * 10 ** (sigma_s * theta_shadow[i, j] / 10)
            beta[i, j] = -34.53 - 38 * np.log10(distance_matrix[i, j]) + 300

    # pilot grouping
    tau_p = 16  # number of pilot symbols index=[0,1,2,3,4]
    pilot_index = np.zeros([1, m_])
    for i in range(m_):
        pilot_index[0, i] = np.random.randint(0, tau_p - 1)

    # calculate alpha
    alpha = np.zeros([m_, n_])
    for i in range(m_):
        same_pilot_current_user = []
        for i_ in range(m_):
            if pilot_index[0, i] == pilot_index[0, i_] and not (i == i_):
                same_pilot_current_user.append(i_)
        for j in range(n_):
            alpha[i, j] = (P_max * tau_p * beta[i, j] ** 2) / (
                    1 + P_max * tau_p * sum(beta[i_, j] for i_ in same_pilot_current_user))

    # calculate snr
    sinr = np.zeros([m_])
    r = np.zeros([m_])
    for i in range(m_):
        for j in range(n_):
            # calculate mu - eq. (7) in ''Performance of Cell-Free Massive MIMO Systems
            # with MMSE and LSFD Receivers''
            same_pilot_current_user = []
            mu_current_user = np.zeros([1, n_])
            for i_ in range(m_):
                if pilot_index[0, i] == pilot_index[0, i_] and not (i == i_):
                    same_pilot_current_user.append(i_)

            for j_ in range(n_):
                mu_current_user[0, j_] = (P_max * tau_p * beta[i, j_] * beta[i, j_]) / (
                        1 + P_max * tau_p * sum(beta[i_, j_] for i_ in same_pilot_current_user))

            mu = np.zeros([m_, n_])
            for i_ in same_pilot_current_user:
                same_pilot = []
                for _ in range(m_):
                    if pilot_index[0, _] == pilot_index[0, i_] and not (_ == i_):
                        same_pilot.append(_)
                for j_ in range(n_):
                    mu[i_, j_] = (P_max * tau_p * beta[i, j_] * beta[i_, j_]) / (
                            1 + P_max * tau_p * sum(beta[_, j_] for _ in same_pilot))
            # calculate Lambda
            Lambda = np.zeros([n_])
            for j in range(n_):
                Lambda[j] = sum(p_[i_] * alpha[i_, j] * beta[i_, j] for i_ in range(m_)) + alpha[i, j]
            Lambda = np.diag(Lambda)
            inv_sum = np.linalg.inv((sum(p_[i_] * np.dot(mu[i_, :].conj().T, mu[i_, :]) for i_ in
                                         same_pilot_current_user) + Lambda))

            sinr[i] = p_[i] * np.dot(np.dot(mu_current_user, inv_sum), mu_current_user.conj().T)
            r[i] = w * math.log2(1 + sinr[i])

    for i in range(m_):
        if r[i] < 1.5e6:
            r[i] = 1.5e6

    return r

def rate_calculate_old(p_current):
    loaded_beta = np.load('sinr_data/beta.npz', 'rb')
    beta = (loaded_beta['arr_0'])
    loaded_gamma = np.load('sinr_data/gamma.npz', 'rb')
    gamma = (loaded_gamma['arr_0'])
    loaded_lsfd = np.load('sinr_data/lsfd_vec.npz', 'rb')
    lsfd_vec = (loaded_lsfd['arr_0'])

    # SINR and transmit rate calculation
    tran_rate = np.zeros([1, M])
    # rate_single = np.zeros([1, M])
    sinr = np.zeros([1, M])
    # sinr_single = np.zeros([1, M])
    bandwidth = 20e6  # system bandwidth in Hz
    noise_power = 3.9810717055349565e-21  # noise power bandwidth per Hz
    for i in range(M):
        sinr_2 = 0
        sinr_1 = p_current[i] * (np.sum(lsfd_vec[i, :] * gamma[i, :]) ** 2)
        for k in range(M):
            sinr_2 = sinr_2 + np.sum(lsfd_vec[k, :] ** 2 * gamma[k, :] * beta[k, :]) * P_t
        sinr_2 = sinr_2 + np.sum(lsfd_vec[i, :] ** 2 * gamma[i, :])
        sinr[0, i] = sinr_1 / sinr_2
        tran_rate[0, i] = bandwidth * math.log2(1 + sinr[0, i])
        # single schemes
        # sinr_single[0, i] = p_current[i] * G_single[0, i] / (
        #         bandwidth * noise_power + P_t * np.sum(G_single))
        # rate_single[0, i] = bandwidth * math.log2(1 + sinr_single[0, i])

    r_array = np.zeros(M)
    for i in range(M):
        r_array[i] = tran_rate[0, i]

    for i in range(M):
        if r_array[i] < 1.5e6:
            r_array[i] = 1.5e6
    return r_array  # M-d array


class UCMEC(gym.Env):
    def __init__(self, render: bool = False):
        # parameter init
        self._render = render
        self.episode_length = 1000
        # action space: [a_1,a_2,...,a_M,p_1,p_2,...,p_M,c_1,c_2,...,c_M]  1x3M continuous vector.
        # a in [0,1], p in [p_min, p_max]  c in [0, c_max]

        # action space normalization
        # p -> [p_min,p_max] -> [p_min-((p_max - p_min)/2), p_max -((p_max - p_min)/2)] -> [-1,1]
        p_high = np.ones(M) * P_max / 0.5
        action_high = p_high
        action_low = -action_high

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(M,), dtype=np.float32)
        # state space: [r_1,r_2,...,r_M]  1xM continuous vector.
        # r in [0, 10000000]
        r_low = np.zeros(M)
        r_high = np.ones(M) * 1e8
        self.observation_space = spaces.Box(low=r_low, high=r_high, shape=(M,), dtype=np.float32)
        self.np_random = None
        self.seed()
        self.step_num = 0

    def step(self, action):
        task_size = np.random.uniform(100000, 200000, [1, M])  # task size in bit
        task_density = np.random.uniform(1000, 1800, [1, M])  # task density cpu cycles per bit
        task_max_delay = np.random.uniform(0.8, 1.2, [1, M])  # task max delay in second
        self.step_num += 1
        # update action and state
        # print("action:",action)
        a_current = [random.random() for i in range(M)]
        # print(a_current)
        p_current = np.clip(action[0:M], -P_max / 2, P_max / 2)
        p_current = p_current + P_max / 2
        _c_current = cp.Variable(M)
        # c_current = c_current + C_edge / 2
        # print("action:", a_current, p_current, c_current)
        state = rate_calculate_old(p_current)
        # state = rate_calculate(M, N, p_current)  # Transmit Rate calculation 1xM
        tran_rate = state  # Transmit Rate
        # print("state:", tran_rate)

        # reward calculation
        # local computing delay
        local_com_delay = []
        for i in range(M):
            local_com_delay.append((1 - a_current[i]) * task_size[0, i] * task_density[0, i] / C_user[0, i])

        # edge offloading delay
        edge_tra_delay = []
        task_list = []
        for i in range(M):

            edge_tra_delay.append(a_current[i] * task_size[0, i] / (tran_rate[i] + 0.001))
            task_list.append(a_current[i] * task_size[0, i] * task_density[0, i])
        _edge_com_delay = cp.multiply(task_list, cp.inv_pos(_c_current * 1e9))
        # total delay
        _total_delay = cp.maximum(local_com_delay, _edge_com_delay + edge_tra_delay)

        # # local computing energy consumption
        # local_com_energy = np.zeros([1, M])
        # for i in range(M):
        #     local_com_energy[0, i] = eta * (1 - a_current[i]) * task_size[0, i] * task_density[0, i] * C_user[0, i] ** 2
        #
        # # edge computing energy consumption & total energy
        # edge_com_energy = np.zeros([1, M])
        # total_energy = np.zeros([1, M])
        # for i in range(M):
        #     # edge computing + transmission energy consumption
        #     edge_com_energy[0, i] = N * p_current[i] * edge_tra_delay[0, i] + P_idle * edge_com_delay[0, i]
        #     total_energy[0, i] = local_com_energy[0, i] + edge_com_energy[0, i]

        # # compare delay with task max delay constraint
        # penalty_sum = 0  # total
        # penalty = 5  # per user
        # success_com_count = np.zeros([1, M])
        # success_tran_count = np.zeros([1, M])
        # success_count = np.zeros([1, M])
        # for i in range(M):
        #     if total_delay[0, i] >= task_max_delay[0, i]:
        #         penalty_sum = penalty_sum + penalty
        #     else:
        #         success_com_count[0, i] = 1
        #
        #     if tran_rate[i] >= sinr_limit:
        #         success_tran_count[0, i] = 1
        #
        #     if success_tran_count[0, i] == 1 and success_com_count[0, i] == 1:
        #         success_count[0, i] = 1

        # # successful probability of computing or transmission
        # success_com_pro = np.sum(success_com_count) / M
        # success_tran_pro = np.sum(success_tran_count) / M
        # success_pro = np.sum(success_count) / M
        #
        # # edge computing resource constraint
        # com_resource_penalty = abs(C_edge - np.sum(c_current)) * 500000

        func = cp.Minimize(cp.sum(_total_delay) / M)
        cons = [0 <= _c_current, cp.sum(_c_current) * 1e9 <= C_edge]
        prob = cp.Problem(func, cons)
        prob.solve(solver=cp.SCS, verbose=False)
        c_current = _c_current.value

        # actual edge offloading delay
        edge_com_delay = []
        for i in range(M):
            edge_com_delay.append(a_current[i] * task_size[0, i] * task_density[0, i] / (c_current[i] * 1e9 + 0.001))

        # actual total delay
        total_delay = np.zeros([1, M])
        for i in range(M):
            total_delay[0, i] = max(local_com_delay[i], edge_com_delay[i] + edge_tra_delay[i])

        reward = -(np.sum(total_delay) / M)
        # print("reward:", reward)
        if self.step_num == self.episode_length:
            done = True
            self.step_num = 0
        else:
            done = False

        # print("average local_com_delay:", np.sum(local_com_delay) / M)
        # print("average edge_com_delay:", np.sum(edge_com_delay) / M)
        # print("average edge_tra_delay:", np.sum(edge_tra_delay) / M)
        # print("average uplink rate:", np.sum(tran_rate) / 10e6 / M)  # Mbit / s
        info = {}
        '''
        print("local_com_energy:", local_com_energy)
        print("edge_com_energy:", edge_com_energy)
        print("total_energy:", np.sum(total_energy))


        print("total_delay:", np.sum(total_delay)/M)
        print("com_resource_penalty:", com_resource_penalty)

        print("state:", tran_rate)
        print("a_current:", a_current)
        print("p_current:", p_current)
        print("c_current:", c_current)

        print("success_com_pro:", success_com_pro)
        print("success_tran_pro:", success_tran_pro)
        print("success_pro:", success_pro)
        '''
        return state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        reset_state = np.random.uniform(low=0, high=1e6, size=(M,))
        return np.array(reset_state)


def evaluate_policy(args, env, agent, state_norm):
    eval_episode = 5
    evaluate_reward = 0
    for eval_episode_index in range(eval_episode):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_

        evaluate_reward += episode_reward

    return evaluate_reward / eval_episode


def main(args, env_name, number, seed):
    start = time.time()
    env = UCMEC(render=False)
    env_evaluate = UCMEC(render=False)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(37)
    env_evaluate.action_space.seed(37)
    np.random.seed(seed)
    torch.manual_seed(seed)
    episode_rewards = []
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = 1000  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))
    plot_int = 1
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    time_list = []
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(
        log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    reward_list = []
    evaluate_reward_list = []
    episode_index = 0
    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        done = False
        cumulative_reward = 0  # Initialize cumulative reward for this episode
        while not done:
            end = time.time()
            runTime = end - start
            runtime_list.append(runTime)
            # print(runTime)
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            ori_r = r
            # print("Episode:", episode_index, "Step:", total_steps, "Reward:", ori_r)
            # print("reward", r)
            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_index != args.max_episode_steps:
                dw = True
                reward_list.append(ori_r)
            else:
                dw = False
            cumulative_reward += ori_r  # Update cumulative reward
            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1
            # print("total_steps", total_steps)

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps % args.evaluate_freq == 0:
            #     evaluate_num += 1
            #     evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
            #     evaluate_rewards.append(evaluate_reward)
            #     print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            #     evaluate_reward_list.append(evaluate_reward)
            #     writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
            #     # Save the rewards
            #     if evaluate_num % args.save_freq == 0:
            #         np.save(
            #             './data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name,
            #                                                                                  number, seed),
            #             np.array(evaluate_rewards))
        episode_rewards.append(cumulative_reward / env.episode_length)
        # Print the rewards every episode
        print("Episode:", episode_index, "Average Reward:", episode_rewards[episode_index])
        episode_index += 1
    save_fn = 'reward_proposed.mat'
    sio.savemat(save_fn, {'reward_proposed': reward_list})
    save_fn = 'evaluate_reward_proposed.mat'
    sio.savemat(save_fn, {'evaluate_reward_proposed': evaluate_reward_list})

    reward_list = loadmat('./reward_proposed.mat')
    reward_list = reward_list['reward_proposed']
    rolling_intv = 60
    df = pd.DataFrame(list(*reward_list))

    d = list(np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
    save_fn = 'reward_proposed_smooth.mat'
    sio.savemat(save_fn, {'reward_proposed_smooth': d})


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(200e3), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=20e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")

    args = parser.parse_args()

    env_name = ['UCMEC-Random-Offloading']
    env_index = 0
    main(args, env_name=env_name[env_index], number=10, seed=20)

    save_fn = 'runtime_list.mat'
    sio.savemat(save_fn, {'runtime_list': runtime_list})
