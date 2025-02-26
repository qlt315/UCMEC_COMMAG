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
import matplotlib.pyplot as plt
runtime_list = []

# Initialization
M = 20  # number of users
N = 50  # number of APs
P_max = 1.3  # maximum transmit power of user / pilot power
P_min = 0  # minimum transmit power of user
P_t = 1  # test transmit power of user
sinr_limit = 5000000

# locations are randomly sampling from 1000x1000 m^2 square area
# locations_users = np.random.random_sample([M, 2]) * 1000  # 2-D location of users
# locations_aps = np.random.random_sample([N, 2]) * 1000  # 2-D location of APs
loaded_locations_users = np.load('SINR_data/locations_users.npz', 'rb')
locations_users = (loaded_locations_users['arr_0'])
loaded_locations_aps = np.load('SINR_data/locations_aps.npz', 'rb')
locations_aps = (loaded_locations_aps['arr_0'])

# locations of single BS
# locations_BS = np.random.random_sample([1, 2]) * 1000
loaded_locations_BS = np.load('SINR_data/locations_aps.npz', 'rb')
locations_BS = (loaded_locations_BS['arr_0'])

# calculate distance between APs and users MxN matrix
distance_matrix = np.zeros([M, N])
distance_matrix_single = np.zeros([1, M])
for i in range(M):
    for j in range(N):
        distance_matrix[i, j] = math.sqrt((locations_users[i, 0] - locations_aps[j, 0]) ** 2
                                          + (locations_users[i, 1] - locations_aps[j, 1]) ** 2)

for i in range(M):
    distance_matrix_single[0, i] = math.sqrt((locations_users[i, 0] - locations_BS[0, 0]) ** 2
                                             + (locations_users[i, 1] - locations_BS[0, 1]) ** 2)

# edge computing parameter
# users parameter
P_idle = 0.3
C_user = np.random.uniform(2e9, 5e9, [1, M])  # computing resource of users  in Hz
Task_size = np.random.uniform(100000, 200000, [1, M])  # task size in bit
Task_density = np.random.uniform(10000, 18000, [1, M])  # task density cpu cycles per bit
Task_max_delay = np.random.uniform(0.8, 1.2, [1, M])  # task max delay in second
eta = 10e-27  # effective switched capacitance

# edge server parameter
# C_edge = np.random.uniform(5, 20)  # computing resource of edge server in Ghz
C_edge = 3.7


class UCMEC(gym.Env):
    def __init__(self, render: bool = False):
        # paremeter init
        self._render = render
        # action space: [a_1,a_2,...,a_M,p_1,p_2,...,p_M,c_1,c_2,...,c_M]  1x3M continuous vector.
        # a in [0,1], p in [p_min, p_max]  c in [0, c_max]

        # action space normalization
        # a -> [0,1]  -> [-0.5,0.5] -> [-1,1]
        # p -> [p_min,p_max] -> [p_min-((p_max - p_min)/2), p_max -((p_max - p_min)/2)] -> [-1,1]
        # c -> [0,c_edge] -> [-c_edge/2, c_edge/2] -> [-1,1]
        a_high = np.ones(M) * 0.5
        p_high = np.ones(M) * P_max / 2
        c_high = np.ones(M) * C_edge / 2
        action_high = np.append(a_high, p_high)
        action_high = np.append(action_high, c_high)
        action_low = -action_high

        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(3 * M,), dtype=np.float32)
        # state space: [r_1,r_2,...,r_M]  1xM continuous vector.
        # r in [0, 10000000]
        r_low = np.zeros(M)
        r_high = np.ones(M) * 1e8
        self.observation_space = spaces.Box(low=r_low, high=r_high, shape=(M,), dtype=np.float32)
        self.np_random = None
        self.seed()
        self.step_num = 0

    def rate_calculation(self, p_current):
        loaded_beta = np.load('SINR_data/beta.npz', 'rb')
        beta = (loaded_beta['arr_0'])
        loaded_gamma = np.load('SINR_data/gamma.npz', 'rb')
        gamma = (loaded_gamma['arr_0'])
        loaded_lsfd = np.load('SINR_data/gamma.npz', 'rb')
        LSFD_vec = (loaded_lsfd['arr_0'])
        loaded_g_single = np.load('SINR_data/g_single.npz', 'rb')
        G_single = (loaded_g_single['arr_0'])

        # SINR and transmit rate calculation
        R = np.zeros([1, M])
        R_single = np.zeros([1, M])
        SINR = np.zeros([1, M])
        SINR_single = np.zeros([1, M])
        W = 20 * math.pow(10, 6)  # system bandwidth in Hz
        noise_power = 3.9810717055349565e-21  # noise power W per Hz
        for i in range(M):
            SINR_2 = 0
            SINR_1 = p_current[i] * (np.sum(LSFD_vec[i, :] * gamma[i, :]) ** 2)
            for k in range(M):
                SINR_2 = SINR_2 + np.sum(LSFD_vec[k, :] ** 2 * gamma[k, :] * beta[k, :]) * P_t
            SINR_2 = SINR_2 + np.sum(LSFD_vec[i, :] ** 2 * gamma[i, :])
            SINR[0, i] = SINR_1 / SINR_2
            R[0, i] = W * math.log2(1 + SINR[0, i])
            # single schemes
            SINR_single[0, i] = p_current[i] * G_single[0, i] / (
                    W * noise_power + P_t * np.sum(G_single))
            R_single[0, i] = W * math.log2(1 + SINR_single[0, i])

        R_array = np.zeros(M)
        for i in range(M):
            R_array[i] = R[0, i] / 10
        return R_array  # M-d array

    def step(self, action):
        self.step_num += 1
        # update action and state
        # print("action:",action)
        a_current = np.clip(action[0:M], -0.5, 0.5)
        p_current = np.clip(action[M:2 * M], -P_max / 2, P_max / 2)
        c_current = np.clip(action[-M:], -C_edge / 2, C_edge / 2)
        a_current = a_current + 0.5
        p_current = p_current + P_max / 2
        c_current = c_current + C_edge / 2
        # print("action:", a_current, p_current, c_current)
        state = self.rate_calculation(p_current)  # Transmit Rate calculation 1xM
        Tran_rate = state  # Transmit Rate
        # print("state:", Tran_rate)

        # reward calculation
        # local computing delay
        local_com_delay = np.zeros([1, M])
        for i in range(M):
            local_com_delay[0, i] = (1 - a_current[i]) * Task_size[0, i] * Task_density[0, i] / C_user[0, i]

        # edge offloading delay
        edge_tra_delay = np.zeros([1, M])
        edge_com_delay = np.zeros([1, M])
        for i in range(M):
            edge_tra_delay[0, i] = a_current[i] * Task_size[0, i] / (Tran_rate[i] + 0.001)
            edge_com_delay[0, i] = a_current[i] * Task_size[0, i] * Task_density[0, i] / (c_current[i] * 1e9 + 0.001)

        # total delay
        total_delay = np.zeros([1, M])
        for i in range(M):
            total_delay[0, i] = max(local_com_delay[0, i], edge_com_delay[0, i] + edge_tra_delay[0, i])

        # local computing energy consumption
        local_com_energy = np.zeros([1, M])
        for i in range(M):
            local_com_energy[0, i] = eta * (1 - a_current[i]) * Task_size[0, i] * Task_density[0, i] * C_user[0, i] ** 2

        # edge computing energy consumption & total energy
        edge_com_energy = np.zeros([1, M])
        total_energy = np.zeros([1, M])
        for i in range(M):
            # edge computing + transmission energy consumption
            edge_com_energy[0, i] = N * p_current[i] * edge_tra_delay[0, i] + P_idle * edge_com_delay[0, i]
            total_energy[0, i] = local_com_energy[0, i] + edge_com_energy[0, i]

        # compare delay with task max delay  delay constraint
        delay_penalty_sum = 0  # total
        penalty = 5  # per user
        success_com_count = np.zeros([1, M])
        success_tran_count = np.zeros([1, M])
        success_count = np.zeros([1, M])
        for i in range(M):
            if total_delay[0, i] >= Task_max_delay[0, i]:
                penalty_sum = delay_penalty_sum + penalty
            else:
                success_com_count[0, i] = 1

            if Tran_rate[i] >= sinr_limit:
                success_tran_count[0, i] = 1

            if success_tran_count[0, i] == 1 and success_com_count[0, i] == 1:
                success_count[0, i] = 1

        # successful probability of computing or transmission
        success_com_pro = np.sum(success_com_count) / M
        success_tran_pro = np.sum(success_tran_count) / M
        success_pro = np.sum(success_count) / M

        # edge computing resource constraint
        com_resource_penalty = abs(C_edge - np.sum(c_current))

        reward = 4000 -(np.sum(total_energy)) -  com_resource_penalty - delay_penalty_sum
        print(reward)

        if self.step_num > 36000:
            done = True
        else:
            done = False

        info = {}
        '''
        print("local_com_energy:", local_com_energy)
        print("edge_com_energy:", edge_com_energy)
        print("total_energy:", np.sum(total_energy))

        print("local_com_delay:", local_com_delay)
        print("edge_com_delay:", edge_com_delay)
        print("edge_tra_delay:", edge_tra_delay)
        print("total_delay:", np.sum(total_delay)/M)
        print("com_resource_penalty:", com_resource_penalty)

        print("state:", Tran_rate)
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
    times = 3
    evaluate_reward = 0
    for _ in range(times):
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

    return evaluate_reward / times


def main(args, env_name, number, seed):
    start = time.time()
    env = UCMEC(render=False)
    env_evaluate = UCMEC(render=False)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = 100  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

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
    while total_steps < args.max_train_steps:

        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            end = time.time()
            runTime = end - start
            runtime_list.append(runTime)
            # print(runTime)
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            # print("reward", r)
            reward_list.append(r)
            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

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
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                evaluate_reward_list.append(evaluate_reward)
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save(
                        './data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name,
                                                                                             number, seed),
                        np.array(evaluate_rewards))

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
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
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

    env_name = ['UCMEC']
    env_index = 0
    main(args, env_name=env_name[env_index], number=10, seed=10)

    save_fn = 'runtime_list.mat'
    sio.savemat(save_fn, {'runtime_list': runtime_list})
