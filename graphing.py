import sys
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np

# used to generate Poisson samples for r_m and m_m
G = np.random.default_rng()

class Opt:
    ''' Class for all variables needed for energy consumption calculation'''
    def __init__(self, S, S_c, r_m, m_m, P_hit):
        self.S = S    		# number of surrogate servers
        self.S_c = S_c      # % cache size (surrogate servers storage capacity)
        self.M = 1000		# total number of contents in the data set
        self.B = 1e6                # size of each content
        self.t = 6000               # time period of the analysis
        self.n_m = self.S_c * S     # number of replicas for content m
        self.r_m = r_m              # requests for content m
        self.m_m = m_m              # modifications to content m
        self.H_A = 3        # hops to fetch content from the same Tier 3 ISP
        self.H_B = 14       # hops to fetch content from the same Tier 2 ISP
        self.H_C = 25               # hops to fetch content from the core network
        self.H_ps = self.H_C        # hops from primary to surrogate servers
        self.T_3 = 1000             # number of Tier 3 ISPs
        self.g_3 = 20       # number of Tier 3 ISPs connected to Tier 2 ISPs
        self.T_2 = 50       # number of Tier 2 ISPs
        self.N = 2e6                # total number of end users
        self.n = 2000               # number of end users per Tier 3 ISP
        self.P_st = 7.84e-12        # storage power consumption per bit
        self.E_r = 1.2e-8           # router energy consumption per bit
        self.E_l = 1.48e-9          # link energy consumption per bit
        self.E_sr = 2.81e-7         # server energy consumption per bit
        self.P_hit = P_hit          # hit rate for content m

def set_opt(S):
    ''' Collects command line args and returns all needed information'''
    if len(sys.argv) == 6:
        S_c = float(sys.argv[1])
        m_m = int(sys.argv[2])
        r_m = int(sys.argv[3])
        P_hit = float(sys.argv[4])
        opt = Opt(S, S_c, r_m, m_m, P_hit)
        return opt
    else:
        print('Usage: python graphing.py <S_c> <r_m> <m_m> <P_hit>')
        sys.exit(1)

def E_storage(opt):
    ''' Calculates CDN storage energy consumption, in Joules'''

    totJ = 0                
    for _ in range(opt.M):
        totJ += opt.B * opt.n_m * opt.P_st * opt.t
    return totJ

def E_server(opt):
    ''' Calculates CDN server energy consumption, in Joules'''
    
    totJ = 0               
    for _ in range(opt.M):
        r_m = G.poisson(opt.r_m)
        totJ += opt.B * r_m * opt.E_sr
    return totJ

def E_synch(opt):
    ''' Calculates CDN synchronization energy consumption, in Joules'''
    
    totJ = 0             
    for _ in range(opt.M):
        lr_energy = opt.E_r * (opt.H_ps + 1) + opt.E_l * (opt.H_ps)
        m_m = G.poisson(opt.m_m)
        totJ += opt.B * m_m * opt.n_m * lr_energy
    return totJ

def E_tran(opt):
    ''' Calculates CDN transmission energy consumption, in Joules'''
    totJ = 0

    P_A = (opt.S * opt.P_hit) / (opt.T_2 * opt.g_3)
    P_B = (opt.S / opt.T_2) * (1 - 1 / opt.g_3) * opt.P_hit
    P_C = 1 - (P_A + P_B)

    for _ in range(opt.M):
        r_m = G.poisson(opt.r_m)
        a_energy = opt.E_r * (opt.H_A + 1) + opt.E_l * (opt.H_A)
        totJ += P_A * opt.B * r_m * a_energy
        b_energy = opt.E_r * (opt.H_B + 1) + opt.E_l * (opt.H_B)
        totJ += P_B * opt.B * r_m * b_energy
        c_energy = opt.E_r * (opt.H_C + 1) + opt.E_l * (opt.H_C)
        totJ += P_C * opt.B * r_m * c_energy

    return totJ
    return 0

def total_energy(opt):
    ''' Calculates the energy consumption of the network, in Joules'''
    return E_server(opt) + E_storage(opt) + E_synch(opt) + E_tran(opt)

def main():

    # values of surrogate servers to test on
    S_lst = [1, 2, 3, 5, 8, 10, 20]

    E_tot_lst = []
    E_syncless_lst = []
    opt = set_opt(S_lst[0])
    for S in S_lst:
        print(S)
        opt = set_opt(S)
        E_tot = total_energy(opt)
        E_syncless = E_tot - E_synch(opt)
        E_tot_lst.append(E_tot)
        E_syncless_lst.append(E_syncless)

    # normalize both E_tot and E_tot-synch lists
    initE = E_tot_lst[0]
    E_tot_norm = [E_tot_lst[i] / initE for i in range(len(E_tot_lst))]
    initE = E_syncless_lst[0]
    E_syncless_norm = [E_syncless_lst[i] / initE for i in range(len(E_syncless_lst))]

    plt.plot(S_lst, E_tot_norm, label = "E_tot", marker="x")
    plt.plot(S_lst, E_syncless_norm, label = "E_tot - synch", marker="o")

    plt.xlabel('Surrogate Servers')
    plt.ylabel('Normalized Energy Consumption')

    # customize the bounds of the x, y axes, and also where ticks are placed
    # plt.xlim((1, 20))
    # plt.ylim((0.9,1.1))
    # plt.xticks(S_lst)
    # plt.yticks([0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
    # plt.yticks([0.9, 0.95, 1.0, 1.05, 1.1])

    plt.title(f'm_m/r_m = {(opt.m_m / opt.r_m):.3f}')
    plt.legend()
    try:
        plt.savefig(sys.argv[5])
    except:
        print(f"Usage: python graphing.py <S_c> <r_m> <m_m> <P_hit> <out.png>")
    
    # analysis printouts --- give us more exact information
    print(f"E_tot: {E_tot_lst}")
    print(f"E_tot-synch: {E_syncless_lst}")

if __name__ == '__main__':
    main()