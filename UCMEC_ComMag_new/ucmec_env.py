import gym
from gym import spaces
from gym.utils import seeding
from os import path
import random
import numpy as np
import pandas as pd
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Initialization
M = 40  # number of users
N = 100  # number of APs
P_max = 1.3  # maximum transmit power of user / pilot power
P_t = 1  # test transmit power of user

# locations are randomly sampling from 1000x1000 m^2 square area
locations_users = np.random.random_sample([M, 2]) * 1000  # 2-D location of users
locations_aps = np.random.random_sample([N, 2]) * 1000  # 2-D location of APs

# locations of single BS
locations_BS = np.random.random_sample([1, 2]) * 1000

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
Task_max_delay = np.random.uniform(2, 5, [1, M])  # task max delay in second
eta = 10e-27  # effective switched capacitance

# edge server parameter
C_edge = np.random.uniform(5e9, 20e9)  # computing resource of edge server


class UCMEC(gym.Env):
    def __init__(self, render: bool = False):
        # paremeter init
        self._render = render
        # action space: [a_1,a_2,...,a_M,p_1,p_2,...,p_M,c_1,c_2,...,c_M]  1x3M continuous vector.
        # a in [0,1], p in [p_min, p_max]  c in [0, c_max]
        a_low = np.zeros(M)
        p_low = np.zeros(M)
        c_low = np.zeros(M)
        a_high = np.ones(M)
        p_high = np.ones(M) * P_max
        c_high = np.ones(M) * C_edge
        action_low = np.append(a_low, p_low)
        action_low = np.append(action_low, c_low)
        action_high = np.append(a_high, p_high)
        action_high = np.append(action_high, c_high)
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
        # shadow loss calculation
        sigma_s = 8  # dB
        a_shadow = np.random.rand(1, N)
        b_shadow = np.random.rand(1, M)
        delta_shadow = random.random()  # random generate form [0,1]
        theta_shadow = np.zeros([M, N])
        for i in range(M):
            for j in range(N):
                theta_shadow[i, j] = math.sqrt(delta_shadow) * a_shadow[0, j] + math.sqrt(1 - delta_shadow) * b_shadow[
                    0, i]  # MxN matrix

        # three slope pass loss calculation
        f_carrier = 1.9 * math.pow(10, 9)  # carrier frequency in Hz
        h_ap = 15  # antenna height of AP
        h_user = 1.65  # antenna height of user
        L = 46.3 + 33.9 * np.log10(f_carrier) - 13.82 * np.log10(h_ap) - (
                1.11 * np.log10(f_carrier) - 0.7) * h_user + 1.56 * np.log10(f_carrier) - 0.8
        d_0 = 10
        d_1 = 50
        mu = np.zeros([M, N])
        beta = np.zeros([M, N])
        gamma = np.zeros([M, N])

        for i in range(M):
            for j in range(N):
                if distance_matrix[i, j] > d_1:
                    mu[i, j] = -L - 35 * np.log10(distance_matrix[i, j])
                elif d_0 <= distance_matrix[i, j] <= d_1:
                    mu[i, j] = -L - 10 * np.log10(d_1 ** 1.5 * (distance_matrix[i, j]) ** 2)
                else:
                    mu[i, j] = -L - 10 * np.log10(d_1 ** 1.5 * d_0 ** 2)

                # beta[i, j] = 10 ** (mu[i, j] / 10) * 10 ** (sigma_s * theta_shadow[i, j] / 10)
                F = np.random.randn(1) * 300
                beta[i, j] = -34.53 - 38 * np.log10(distance_matrix[i, j]) + 300
                gamma[i, j] = (M * P_max * beta[i, j] ** 2) / (1 + M * P_max * beta[i, j])

        # pilot grouping
        tau_p = 5  # number of pilot symbols index=[0,1,2,3,4]
        pilot_index = np.zeros([1, M])
        for i in range(M):
            pilot_index[0, i] = np.random.randint(0, tau_p - 1)

        # LSFD calculation
        LSFD_vec = np.zeros([M, N])  # for each user, the size of LSFD vector is 1xN
        gamma_lsfd_sum = 0  # represent the sum of gamma_m * gamma_m^{H}, which is a number
        Lambda = np.zeros([N, N])  # represent diag(...), which is a NxN diag matrix
        for i in range(M):
            Lambda_sum = np.zeros([1, N])  # represent the sum of gamma_m * beta_m, which is a 1xN vector
            for k in range(M):
                Lambda_sum = Lambda_sum + gamma[k, :] * beta[k, :]
            Lambda_element = P_max * Lambda_sum + gamma[i, :]
            for j in range(N):
                Lambda[j, j] = Lambda_element[0, j]
            for k in range(M):
                if k != i and pilot_index[0, k] == pilot_index[0, i]:
                    gamma_lsfd_sum = gamma_lsfd_sum + np.dot(gamma[k, :],
                                                             np.transpose(gamma[k, :]))  # 1xN * (1xN)^T = 1x1

            # print(P_max * gamma_lsfd_sum * np.ones([N, N]) + Lambda)
            LSFD_vec[i, :] = np.transpose(np.dot(np.linalg.inv(P_max * gamma_lsfd_sum * np.ones([N, N]) + Lambda)
                                                 , np.transpose(gamma[i, :])))

        # channel gain of single schemes calculation
        G_single = np.zeros([1, M])
        h_single = np.zeros([1, M])
        for i in range(M):
            G_single[0, i] = distance_matrix_single[0, i] ** (-4)
            h_single[0, i] = np.random.randn(1)
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

        return np.array(R)  # M-d array

    def step(self, action):
        self.step_num += 1
        # update action and state
        a_current = np.clip(action[0:M], 0, 1)
        p_current = np.clip(action[M:2 * M], 0, P_max)
        c_current = np.clip(action[-M:], 0, C_edge)
        print(p_current)
        state = self.rate_calculation(p_current)  # Transmit Rate calculation 1xM
        Tran_rate = state  # Transmit Rate

        # reward calculation
        # local computing delay
        local_com_delay = np.zeros([1, M])
        for i in range(M):
            local_com_delay[0, i] = (1 - a_current[i]) * Task_size[0, i] * Task_density[0, i] / C_user[0, i]

        # edge offloading delay
        edge_tra_delay = np.zeros([1, M])
        edge_com_delay = np.zeros([1, M])
        for i in range(M):
            edge_tra_delay[0, i] = a_current[i] * Task_size[0, i] / Tran_rate[0, i]
            edge_com_delay[0, i] = a_current[i] * Task_size[0, i] * Task_density[0, i] / c_current[i]

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
            edge_com_energy[0, i] = N * p_current[i] * edge_tra_delay[0, i] + P_idle * edge_com_delay[0, i]
            total_energy[0, i] = local_com_energy[0, i] + edge_com_energy[0, i]

        # compare delay with task max delay
        penalty_sum = 0  # total
        penalty = 5  # per user
        for i in range(M):
            if total_delay[0, i] >= Task_max_delay[0, i]:
                penalty_sum = penalty_sum + penalty

        reward = -(np.sum(total_energy) + penalty_sum)

        if self.step_num > 36000:
            done = True
        else:
            done = False

        info = {}
        return state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        reset_state = np.random.uniform(low=0, high=1e6, size=(M,))
        return np.array(reset_state)


if __name__ == "__main__":
    env = UCMEC(render=False)
    # check_env(env)
    obs = env.reset()
    n_steps = 50
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        print(f"state: {obs} \n")
        print(f"action : {action}, reward : {reward}")