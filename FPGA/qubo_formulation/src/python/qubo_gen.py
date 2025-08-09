
import numpy as np
from itertools import combinations



def build_log_xrate_dict(problem, cur_lst):
    log_xrate_dict = {}

    for fxdata in problem:
        log_xrate_dict[(cur_lst.index(fxdata[0][:3]), cur_lst.index(fxdata[0][4:]))] = -np.log(float(fxdata[-1]))
        log_xrate_dict[(cur_lst.index(fxdata[0][4:]), cur_lst.index(fxdata[0][:3]))] = np.log(float(fxdata[-2]))

    return log_xrate_dict










def build_Q(problem, cur_lst, M1, M2, mode='QUBO'):

    log_xrate_dict = build_log_xrate_dict(problem, cur_lst)

    var_lst = list(log_xrate_dict.keys())
    N = len(var_lst)

    pen1 = np.zeros(N**2).reshape((N, N))
    pen2 = np.zeros(N**2).reshape((N, N))

    for k in range(len(cur_lst)):
        v1 = np.array([(k == i) - (k == j) for i, j in var_lst], dtype=int)
        pen1 += np.outer(v1, v1)

        v2 = np.array([(k == i) for i, j in var_lst], dtype=int)
        pen2 += np.outer(v2, v2) - np.diag(v2)

    if mode == 'QUBO':
        return -np.diag([log_xrate_dict[var] for var in var_lst]) + M1 * pen1 + M2 * pen2
    if mode == 'Ising':
        return qubo2ising(-np.diag([log_xrate_dict[var] for var in var_lst]) + M1 * pen1 + M2 * pen2)








def qubo2ising(Q):

    S = (Q + Q.T) / 2

    J = (S - np.diag(np.diag(S))) / 4
    h = np.sum(S, axis=0) / 2
    offset = (np.sum(S) + np.sum(np.diag(S))) / 4

    return J, h, offset








def ising2qubo(J, h):

    S = (J + J.T) / 2

    Q = 4*J + np.diag(2*h - 4*np.sum(J, axis=0))
    offset = np.sum(J) - np.sum(h)

    return Q, offset



def main():
    cur_lst = ['USD', 'EUR', 'JPY', 'GBP', 'CHF']
    problem = [['EUR/GBP', '20210401 00:00:00.596', '0.85062', '0.85069'],
               ['EUR/JPY', '20210401 00:00:00.991', '129.877', '129.885'],
               ['USD/CHF', '20210401 00:00:00.730', '0.94391', '0.94397'],
               ['EUR/CHF', '20210401 00:00:00.590', '1.10691', '1.107'],
               ['CHF/JPY', '20210401 00:00:00.989', '117.325', '117.337'],
               ['GBP/JPY', '20210401 00:00:00.986', '152.673', '152.691'],
               ['EUR/USD', '20210401 00:00:00.656', '1.17267', '1.17272'],
               ['GBP/USD', '20210401 00:00:00.991', '1.37851', '1.37863'],
               ['USD/JPY', '20210401 00:00:00.995', '110.753', '110.759']]
    print('Sample problem:\n', problem)

    M1 = 50
    M2 = 25
    Q = build_Q(problem, cur_lst, M1, M2)
    print('Q:\n', Q)



if __name__ == '__main__':
    main()
