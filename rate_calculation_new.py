import random
import numpy as np
import math


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
    return r


def cellular_rate_calculate(m_, n_, p_):
    # channel gain of single schemes calculation
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

    g = np.empty([m_, n_], dtype=complex)  # channel coefficient
    h = np.empty([m_, n_], dtype=complex)  # small scale fading ~ CN(0,1)
    for i in range(m_):
        for j in range(n_):
            real_part = np.random.normal(0, 1 / 2, 1)
            imag_part = np.random.normal(0, 1 / 2, 1)
            h[i, j] = np.sqrt(1 / 2) * (real_part[0] + 1j * imag_part[0])
            g[i, j] = np.sqrt(beta[i, j]) * h[i, j]

    # SINR and transmit rate calculation
    r_cellular = np.zeros([m_])
    sinr_cellular = np.zeros([m_])
    w = 20e6  # system bandwidth in Hz
    # OFDMA transmission
    w_per_user = w / m_
    noise_power = 3.9810717055349565e-21  # noise power w per Hz

    # choose the nearest BS to associate
    choose_bs_list = np.zeros(m_)
    for i in range(m_):
        choose_bs_list[i] = np.argmax(np.abs(g[i, :]))  # return the base station index with maximum channel gain

    for i in range(m_):
        # interfere calculation
        interference = 0
        for k in range(m_):
            if k == i:
                pass
            else:
                interference = interference + p_[k] * np.abs(g[k, int(choose_bs_list[k])]) ** 2
        # single scheme sinr calculate
        sinr_cellular[i] = p_[i] * np.abs(g[i, int(choose_bs_list[i])]) ** 2 / (
                w_per_user * noise_power)
        r_cellular[i] = w_per_user * math.log2(1 + sinr_cellular[i])
    return r_cellular


if __name__ == "__main__":
    # Initialization
    M = 20  # number of users
    N = 50  # number of APs
    P_max = 0.30  # maximum transmit power of user / pilot power
    P_t = 1  # test transmit power of user
    np.random.seed(37)
    # locations are randomly sampling from 1000x1000 m^2 square area
    locations_users = np.random.random_sample([M, 2]) * 1000  # 2-D location of users
    locations_aps = np.random.random_sample([N, 2]) * 1000  # 2-D location of APs

    # locations of single BS
    locations_BS = np.random.random_sample([1, 2]) * 1000

    # calculate distance between APs and users MxN matrix
    distance_matrix = np.zeros([M, N])
    distance_matrix_single = np.zeros([1, M])

    # save data
    np.savez('sinr_data/locations_users', locations_users)
    np.savez('sinr_data/locations_aps', locations_aps)
    np.savez('sinr_data/locations_BS', locations_BS)

    for i in range(M):
        for j in range(N):
            distance_matrix[i, j] = math.sqrt((locations_users[i, 0] - locations_aps[j, 0]) ** 2
                                              + (locations_users[i, 1] - locations_aps[j, 1]) ** 2)

    for i in range(M):
        distance_matrix_single[0, i] = math.sqrt((locations_users[i, 0] - locations_BS[0, 0]) ** 2
                                                 + (locations_users[i, 1] - locations_BS[0, 1]) ** 2)
    p = np.ones([M]) * P_max
    R = rate_calculate(M, N, p)
    R_single = cellular_rate_calculate(M, N, p)

    print("Rate:", R / 10e6)
    print("R_single", R_single / 10e6)
    print("Average Rate:", np.sum(R) / 10e6 / M)
    print("Average R_single:", np.sum(R_single) / 10e6 / M)

    # loaded_data = np.load('sinr_data/beta.npz','rb')
    # print(loaded_data['arr_0'])
