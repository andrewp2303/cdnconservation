import sys, os
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.special import comb

# used to generate Poisson samples for r_m and m_m
G = np.random.default_rng()

class Opt:
    ''' 
    Class for all variables needed for energy consumption calculation.

    *Note*: this can be modified to include more accurate data, if looking
    to model how content on a specific site could be optimally distributed 
    using surrogate servers in a CDN.
    '''
    def __init__(self, S):

        # model used (old, new)
        self.model = 'old'

        # variables (set these first)
        self.S = S    		# number of surrogate servers
        self.S_c = 0.4      # % cache size (surrogate storage capacity) (0.2, 0.4, 0.5)
        self.m_m = 10              # modifications to content m (10, 100)
        self.r_m = 10000              # requests for content m (100, 1000, 10000)
        self.P_hit = 0.7847          # hit rate for content m (0.7847)

        # constants / dependent variables
        self.M = 1000		# total number of contents in the data set
        self.B = 1e6                # size of each content
        self.t = 6000               # time period of the analysis
        self.n_m = self.S_c * S     # number of replicas for content m
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

def hgeom_pmf(k, w, b, n):
    ''' PMF using HGeom(w, b, n) applied to k '''

    wchoosek = comb(w,k)
    bchoosenk = comb(b, n-k)
    wbchoosen = comb(w+b,n)
    return (wchoosek * bchoosenk) / wbchoosen

def E_tran(opt):
    ''' Calculates CDN transmission energy consumption, in Joules'''
    totJ = 0

    # probability of earliest hit occurring at Tier 3, 2, 1 respectively

    if opt.model == 'new':
        P_A = (opt.S / opt.T_3) * opt.P_hit
        P_B = sum([hgeom_pmf(i, opt.S, opt.T_3 - opt.S, opt.g_3) * (1 - (1 - opt.P_hit) ** i) for i in range(1, opt.g_3 + 1)]) - P_A
        P_C = 1 - (P_A + P_B)
    else:
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

def total_energy(opt):
    ''' Calculates the energy consumption of the network, in Joules'''
    return E_server(opt) + E_storage(opt) + E_synch(opt) + E_tran(opt)

def main():

    # values of surrogate servers to test on
    # S_lst = [1, 2, 3, 5, 8, 10, 20, 50, 100, 500, 1000]
    # S_lst = [1, 2, 3, 5, 8, 10, 20, 50, 100, 200]
    S_lst = [2 ** i for i in range(10)]

    E_tot_lst = []
    E_syncless_lst = []
    opt = Opt(S_lst[0])
    for S in S_lst:
        opt = Opt(S)
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

    if len(sys.argv) == 2:
        plt.savefig(os.path.join("plots", sys.argv[1]))
    else:
        print(f"Usage: python conservation.py <out.png>")
    
    # analysis printouts --- give us more exact information
    print(f"E_tot: {E_tot_lst}")
    print(f"E_tot-synch: {E_syncless_lst}")

if __name__ == '__main__':
    main()