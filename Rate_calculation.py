import random
import numpy as np
import math

# Initialization
M = 20  # number of users
N = 50  # number of APs
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

# save data
np.savez('SINR_data/locations_users', locations_users)
np.savez('SINR_data/locations_aps', locations_aps)
np.savez('SINR_data/locations_BS', locations_BS)

for i in range(M):
    for j in range(N):
        distance_matrix[i, j] = math.sqrt((locations_users[i, 0] - locations_aps[j, 0]) ** 2
                                          + (locations_users[i, 1] - locations_aps[j, 1]) ** 2)

for i in range(M):
    distance_matrix_single[0, i] = math.sqrt((locations_users[i, 0] - locations_BS[0, 0]) ** 2
                                             + (locations_users[i, 1] - locations_BS[0, 1]) ** 2)


def Rate_calculate(M, N):  # calculate SINR of all users under cell-free MIMO   -> MxN matrix
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
                gamma_lsfd_sum = gamma_lsfd_sum + np.dot(gamma[k, :], np.transpose(gamma[k, :]))  # 1xN * (1xN)^T = 1x1

        # print(P_max * gamma_lsfd_sum * np.ones([N, N]) + Lambda)
        LSFD_vec[i, :] = np.transpose(np.dot(np.linalg.inv(P_max * gamma_lsfd_sum * np.ones([N, N]) + Lambda)
                                             , np.transpose(gamma[i, :])))

    # channel gain of single schemes calculation
    G_single = np.zeros([1, M])
    h_single = np.zeros([1, M])
    for i in range(M):
        G_single[0, i] = distance_matrix_single[0, i] ** (-4)

    # save data
    np.savez('SINR_data/beta', beta)
    np.savez('SINR_data/gamma', gamma)
    np.savez('SINR_data/lsfd_vec', LSFD_vec)
    np.savez('SINR_data/g_single', G_single)

    # SINR and transmit rate calculation
    R = np.zeros([1, M])
    R_single = np.zeros([1, M])
    SINR = np.zeros([1, M])
    SINR_single = np.zeros([1, M])
    W = 20 * math.pow(10, 6)  # system bandwidth in Hz
    noise_power = 3.9810717055349565e-21  # noise power W per Hz
    for i in range(M):
        SINR_2 = 0
        SINR_1 = P_t * (np.sum(LSFD_vec[i, :] * gamma[i, :]) ** 2)
        for k in range(M):
            SINR_2 = SINR_2 + np.sum(LSFD_vec[k, :] ** 2 * gamma[k, :] * beta[k, :]) * P_t
        SINR_2 = SINR_2 + np.sum(LSFD_vec[i, :] ** 2 * gamma[i, :])
        SINR[0, i] = SINR_1 / SINR_2
        R[0, i] = W * math.log2(1 + SINR[0, i])
        # single schemes
        SINR_single[0, i] = P_t * G_single[0, i] / (
                W * noise_power + P_t * np.sum(G_single))
        R_single[0, i] = W * math.log2(1 + SINR_single[0, i])

    return R, R_single


R, R_single = Rate_calculate(M, N)


print(np.sum(200000 * np.ones([1, M]) / R) / M)

print(np.sum(200000 * np.ones([1, M]) / R_single) / M)

# loaded_data = np.load('SINR_data/beta.npz','rb')
# print(loaded_data['arr_0'])
