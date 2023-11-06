import numpy as np
from numpy import linalg
from scipy.linalg import sqrtm
import networkx as nx

import perceval as pcvl
from perceval.algorithm import Sampler
import perceval.components as comp

def random_graph_generation(n_nodes, p1, n_densest=0, p2=0):
    '''Generates a random graph with:
           .n_nodes and edges with probability p1
           .n_densest - selects n_densest nodes to add edges with probability p2
    '''
    G1 = nx.Graph()
    G1.add_nodes_from([k for k in range(n_nodes-n_densest)])

    # adding edges with probability p1 for the given nodes
    for i in range(n_nodes-n_densest):
        for j in range(i+1, n_nodes-n_densest):
            r = np.random.random()
            if r < p1:
                G1.add_edge(i, j)

    # adding edges with probability p2 for selected nodes with probability p2
    G2 = nx.Graph()
    G2.add_nodes_from([k for k in range(n_nodes-n_densest, n_nodes)])

    # adding edges with probability p1 for the given nodes
    for i in range(n_nodes-n_densest, n_nodes):
        for j in range(i+1, n_nodes):
            r = np.random.random()
            if r < p2:
                G2.add_edge(i, j)

    G = nx.compose(G1, G2)
    if n_densest != 0:
        #n_connect=round(min(n_densest, n_nodes-n_densest)/2)+1
        n_connect = round(n_densest/2)+1
        for i in range(n_connect):
            node1 = np.random.randint(n_nodes-n_densest)
            node2 = np.random.randint(n_nodes-n_densest, n_nodes)
            G.add_edge(node1, node2)

    # so there is not isolated parts
    if list(nx.isolates(G)) != [] or len(list(G.subgraph(c) for c in nx.connected_components(G))) > 1:
        G1.clear()
        G2.clear()
        G.clear()
        G = random_graph_generation(n_nodes, p1, n_densest, p2)

    return G


def to_unitary(A):
    ''' Input: graph A either as:
                                 an adjacency matrix of size mxm
                                 a networkX graph with m nodes
        Output: unitary with size 2mx2m
    '''

    if type(A) == type(nx.Graph()):
        A = nx.convert_matrix.to_numpy_matrix(A)
    P, D, V = linalg.svd(A)

    c = np.max(D)
    # if it is not complex, then np.sqrt will output nan in complex values
    An = np.matrix(A/c, dtype=complex)
    P = An
    m = len(An)
    Q = sqrtm(np.identity(m)-np.dot(An, An.conj().T))
    R = sqrtm(np.identity(m)-np.dot(An.conj().T, An))
    S = -An.conj().T
    Ubmat = np.bmat([[P, Q], [R, S]])
    #Ubmat[abs(Ubmat.imag) < 10**(-6)] = 0
    Ubmat = Ubmat.real
    #print(Ubmat)
    return (np.copy(Ubmat), c)


def input_state(m):
    '''input state for selection of our m modes
        returns |1,1,1,...,0,0,0> m ones and m zeros'''
    return np.append(np.ones(m), np.zeros(m)).astype(int)


# Post selection of samples with photons only on first half modes
def post_select(samples):
    ''''post select on states that have all modes from m to 2*m as vacuum
        can't have collision of first half'''
    a = []
    m = int(len(samples[0])/2)
    for state in samples:
        state = list(state)
        if all(ele == state[m-1] for ele in state[:m]) and state[m-1] == 1:
            # do not need to check if there is vaccum in the second half for several reasons!
            a.append(state)
    return a


def perm_estimation(G, nb_samples, Ns_min=0):
    if Ns_min == 0:
        Ns_min = nb_samples

    if type(G) == type(nx.Graph()): 
        m = G.number_of_nodes()
    else:
        m = len(G)
    in_state = input_state(m)

    U, c = to_unitary(G)
    U_matrix_pcvl = pcvl.Matrix(U)
    unitary_component = comp.Unitary(U_matrix_pcvl)
    proc = pcvl.Processor("CliffordClifford2017", unitary_component)
    proc.with_input(pcvl.BasicState( in_state))

    samples = []
    i = 0
    sampler = Sampler(proc)
    while len(samples) < Ns_min:
        samples = sampler.samples(nb_samples)['results']
        samples = post_select(samples)
        i = i+1
    print("Total number of samples: ", nb_samples*i)
    print("Number of samples post:", len(samples))
    perm = (c**m)*np.sqrt(len(samples)/(nb_samples*i))
    return perm

def perm_estimation(G, nb_samples, Ns_min=0):
    if Ns_min == 0:
        Ns_min = nb_samples

    if type(G) == type(nx.Graph()):
        m = G.number_of_nodes()
    else:
        m = len(G)
    in_state = input_state(m)

    U, c = to_unitary(G)
    U_matrix_pcvl = pcvl.Matrix(U)
    unitary_component = comp.Unitary(U_matrix_pcvl)
    proc = pcvl.Processor("CliffordClifford2017", unitary_component)
    proc.with_input(pcvl.BasicState(in_state))

    samples_accepted = []
    i = 0
    sampler = Sampler(proc)
    while len(samples_accepted) < Ns_min:
        samples_accepted.append(list(sampler.samples(nb_samples)['results']))
        samples_accepted = post_select(samples_accepted)
        i = i+1
    print("Total number of samples: ", nb_samples*i)
    print("Number of samples post:", len(samples_accepted))
    perm = (c**m)*np.sqrt(len(samples_accepted)/(nb_samples*i))
    return perm
