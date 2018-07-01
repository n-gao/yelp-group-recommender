import numpy as np
import random
from multiprocessing import Pool
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor
import src.preprocessing as pre

_states = []
_perm_d = None
_M = 0
_g = 0
_state_correction = {}
_adj = None
_steps = 0

def monte_carlo_markov_chain(seed):
    """
        Simulates one markov chain for @steps iterations.
    """
    np.random.seed(seed)
    rand_states = _states[_state_correction[np.random.randint(0, _M, _steps)]]
    P = _states[_state_correction[random.randrange(0, _M)]]
    for Q in rand_states:
    #for i in range(_steps):
        #Q = random.randrange(0, _M)
        #Q = _states[_state_correction[Q]]
        if sum([(P not in p) or (Q in p and (p[Q] < p[P])) for p in _perm_d]) > _g/2:
            P = Q
    return P

def get_adjacency_matrix():
    result = np.zeros((_M, _M), dtype=bool)
    for P in range(_M):
        P_s = _states[_state_correction[P]]
        for Q in range(_M):
            Q_s = _states[_state_correction[Q]]
            if Q != P and sum([(P_s not in p) or (Q_s in p and (p[Q_s] < p[P_s])) for p in _perm_d]) > _g/2:
                result[P, Q] = True
    return result

def get_dead_ends():
    ends = []
    for P in range(_M):
        if is_dead_end(P):
            P_s = _states[_state_correction[P]]
            ends.append(P_s)
    return ends

def is_dead_end(P):
    P_s = _states[_state_correction[P]]
    breaked = False
    for Q in range(_M):
        Q_s = _states[_state_correction[Q]]
        #if Q != P and (sum([(P_s not in p) or (Q_s in p and (p[Q_s] < p[P_s])) for p in _perm_d]) > _g/2) != _adj[P, Q]:
        #    print(P_s, Q_s, P, Q, _adj[P, Q])
        if Q != P and sum([(P_s not in p) or (Q_s in p and (p[Q_s] < p[P_s])) for p in _perm_d]) > _g/2:
            breaked = True
            break
    return not breaked

def simulate_markov_chains(rel_steps, num_iter, permutations, n=10, processes=20, min_samples=100, max_items_per_step=-1, verbose=False):
    """
        Simulates @num_iter Markov Chain processes for @steps steps.
        n defines the number of items which should be returnes.
        permutations is the rankings of shape [N, M], where N is the number of users and M number of businesses
    """
    global _M
    global _perm_d
    global _g
    global _states
    global _state_correction
    global _adj
    global _steps
    _states = np.unique(permutations)
    _M = _states.shape[0]
    _g = permutations.shape[0]
    _removed = []
    
    _state_correction = np.arange(_M)

    _perm_d = []
    for i in range(_g):
        d = {}
        for ind, j in enumerate(permutations[i]):
            d[j] = ind
        _perm_d.append(d)
    
    result = []
    step = 0
    
    #_adj = get_adjacency_matrix()
    while len(result) < n and _M > 0:
        run_results = get_dead_ends()
        # If there are no dead ends simulate
        #run_results = []
        if len(run_results) == 0:
            if verbose:
                print('Simulating ...', end='\r')
            _steps = int(_M * rel_steps)
            with Pool(processes) as p:
                run_results = p.map(monte_carlo_markov_chain, np.random.randint(0, 4294967295, num_iter, dtype=np.uint32))
            _min_samples = min_samples
        
            # Get most common result with min_samples of occurrences
            most_common = Counter(run_results).most_common()
            #print(Counter(run_results).most_common(10))
            i = 0
            run_order = []
            while len(run_order) == 0:
                while i < len(most_common) and most_common[i][1] > _min_samples and (max_items_per_step <= 0 or len(run_order) != max_items_per_step):
                    run_order.append(most_common[i][0])
                    i += 1
                _min_samples = _min_samples * 0.9
        else:
            _min_samples = 0
            run_order = run_results
                
        #order_inds = np.where(np.isin(_states, run_order))[0]
        #inv_correction = {j: i for i, j in enumerate(_state_correction)}
        #inds = [inv_correction[i] for i in order_inds]
        #mask = np.ones(_M, dtype=bool)
        #mask[inds] = False
        #print('Removed: ', inds)
        #_adj = _adj[mask]
        #print('Adj1')
        #print(_adj)
        #_adj = _adj[:,mask]
        #print(_adj)
        
        #run_order = [i[0] for i in Counter(run_results).most_common()]
        result = result + run_order
        _M -= len(run_order)
        
        order_inds = np.where(np.isin(_states, run_order))[0]
        #print(_state_correction, np.isin(_state_correction, order_inds).sum())
        _state_correction = _state_correction[np.invert(np.isin(_state_correction, order_inds))]
        #_state_correction = pre.update_idx(_state_correction, order_inds)
        #print(_state_correction)
        #new_correction = np.zeros(_M, dtype=np.int64)
        #for s in range(_M):
        #    add = sum([i <= s for i in order_inds])
        #    while _state_correction[s + add] in order_inds:
        #        add += 1
        #    new_correction[s] = _state_correction[s + add]
        #_state_correction = new_correction
                
        step += 1
        if verbose:
            print('%i. Simulation ended; Found: %s; Total of %i items' % (step, str(run_order), len(result)))
        
        if n < 0:
            break
    if verbose:
        print('Found %i items. End.' % len(result))
    return result[:n]

def calculate_markov_chain(permutations):
    """
        This method is not working. DO NOT USE!
    """
    states = np.unique(permutations)
    M = states.shape[0]
    g = permutations.shape[0]
    
    perm_d = []
    for i in range(g):
        d = {}
        for ind, j in enumerate(permutations[i]):
            d[j] = ind
        perm_d.append(d)
    
    T = np.zeros((M, M))
    for i, i_s in enumerate(states):
        for j, j_s in enumerate(states):
            if i == j:
                continue
            T[i, j] = 1/M if sum([(i_s not in p) or (j_s in p and (p[i_s] < p[j_s])) for p in perm_d]) > g/2 else 0
        T[i, i] = 1 - T[i].sum()
        
    P_next = np.zeros(M)
    P_next[np.random.randint(0, M)] = 1
    P = np.zeros_like(P_next)
    while not np.allclose(P, P_next):
        P = P_next.copy()
        P_next = P_next.dot(T)
    return P_next, np.sum(P_next), np.max(P_next), np.argmax(P_next), states[np.argmax(P_next)]
