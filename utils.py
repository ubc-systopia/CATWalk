import numpy as np
import torch
import random
import argparse
import sys
import os
from itertools import combinations

def generate_he_info(n_v, ts, v_simplices):
    full_he_info = {} # for each hyperedge: (set(nodes) and timestamp)
    v_start_idx = 0
    for he_idx, (n_v_i, ts_i) in enumerate(zip(n_v, ts)):
        he_i_nodes = set(v_simplices[v_start_idx : v_start_idx + n_v_i])
        v_start_idx += n_v_i

        he_i = (he_i_nodes, ts_i)
        full_he_info[he_idx+1] = he_i
    return full_he_info

def generate_he_info_for_CE(n_v, ts, v_simplices):
    full_he_info = {} # for each hyperedge: (set(nodes) and timestamp)
    v_start_idx = 0
    s_he_idx = 1
    for he_idx, (n_v_i, ts_i) in enumerate(zip(n_v, ts)):
        he_i_nodes = set(v_simplices[v_start_idx : v_start_idx + n_v_i])
        v_start_idx += n_v_i
        #make simple edges from hyperedge
        vertex_pairs = combinations(he_i_nodes, 2)

        for simple_edge in vertex_pairs:
            s_he_i_nodes = set(simple_edge)
            s_he_i = (s_he_i_nodes, ts_i)
            full_he_info[s_he_idx] = s_he_i
            s_he_idx += 1        

    return full_he_info

def convert_strList_to_intList(l):
    return [int(x) for x in l]

def generate_nc_data_structures(hes, node_labels, label_names):
    hes_list = []
    for he in hes:
        he_nodes = convert_strList_to_intList(he.split(","))
        hes_list += [set(he_nodes)]
    
    node_labels_mapping = {}
    for i, label in enumerate(node_labels):
        node_labels_mapping[i+1] = label
    
    label_name_mapping = {}
    for i, name in enumerate(label_names):
        label_name_mapping[i+1] = name
    
    return hes_list, node_labels_mapping, label_name_mapping

    

def build_node_temporal_adjlist(max_node_idx, he_info):
    """
    Params
    ------
    n_nodes: int (number of nodes)
    he_info : { int : (set, int)}  (mapping he_idx : (set(nodes), ts))

    Output
    --------
    n_adj_list: List[List[int]]
    """
    n_adj_list = [[] for _ in range(max_node_idx+1)]
    for he_idx in he_info:
        he_nodes, he_ts = he_info[he_idx]
        for node in he_nodes:
            other_nodes = he_nodes - {node}
            n_adj_list[node].extend([(n, he_idx, he_ts) for n in other_nodes]) 
    return n_adj_list

def process_sampling_numbers(num_neighbors, num_layers):
    if not type(num_neighbors)==list: # handle default value
        num_neighbors = [num_neighbors]
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers

def construct_algo_data_given_he_ids(valid_he_ids, he_info):
    src_l, he_offset_l, ts_l = [], [0], []

    prev_he_offset_val = 0
    for he_idx in valid_he_ids:
        he_nodes = he_info[he_idx][0]
        src_l.extend(list(he_nodes))
        prev_he_offset_val += len(he_nodes)
        he_offset_l.append(prev_he_offset_val)
        ts_l.extend([he_info[he_idx][1]]*len(he_nodes))
    
    return src_l, he_offset_l, ts_l 

def nc_transfer_lr_construct_algo_data_given_nodes(batch_nodes, he_info, train_time):
    src_l, he_offset_l, ts_l = [], [0], []

    prev_he_offset_val = 0
    for node in batch_nodes:
        src_l.append(node)
        prev_he_offset_val += 1
        he_offset_l.append(prev_he_offset_val)
        ts_l.append(train_time)
    
    return src_l, he_offset_l, ts_l 

def construct_algo_data_given_nodes(batch_nodes, he_info, sampled_he_per_node):
    
    sampled_he_idxs = []
    for node in batch_nodes:
        #find hes having node in them
        hes_having_node = []
        for i in he_info:
            if(node in he_info[i][0]):
                hes_having_node += [i]
        sampled_hes_for_node = random.choices(hes_having_node, k=sampled_he_per_node)

        sampled_he_idxs.extend(sampled_hes_for_node)
    
    return construct_algo_data_given_he_ids(sampled_he_idxs, he_info)

class RandHyperEdgeSampler(object):
    def __init__(self, nodes):
        nodes =  set().union(*nodes)
        self.nodes_list = np.array(list(nodes))

    def sample(self, src_l, he_offset_l):
        fake_src_l = []
        src_l = np.array(src_l)
        for he_idx in range(len(he_offset_l)-1):
            s_idx, e_idx = he_offset_l[he_idx], he_offset_l[he_idx+1]
            he_nodes = src_l[s_idx:e_idx]
            he_size = e_idx - s_idx
            remained_nodes = np.setdiff1d(self.nodes_list, he_nodes)

            #keep half of the nodes in the source hyperedge and replace the rest
            n_kept_nodes, n_random_nodes = he_size//2, he_size - he_size//2
            kept_nodes = np.random.choice(he_nodes, n_kept_nodes)
            random_nodes = np.random.choice(remained_nodes, n_random_nodes)

            fake_src_l.extend(kept_nodes)
            fake_src_l.extend(random_nodes)

        return fake_src_l

class EarlyStopMonitor(object):
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_args():
    parser = argparse.ArgumentParser('Interface for CATWALK: Inductive Dynamic Representation Learning for Link Prediction on Temporal Hyper Graphs')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try email-Enron or NDC-classes',
                        choices=['NDC-classes', 'contact-primary-school', 'contact-high-school','email-Enron', 'email-Eu', 'congress-bills', 'tags-math-sx', 
                        'threads-ask-ubuntu', 'NDC-substances', 'tags-ask-ubuntu'],
                        default='NDC-classes')
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')
    parser.add_argument('--pretrained', type=str, default='True', choices=['True', 'False'], help='use pretrained models or not')


    # method-related hyper-parameters
    parser.add_argument('--n_degree', nargs='*', default=6,
                        help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
    parser.add_argument('--n_layer', type=int, default=3, help='number of network layers')
    parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference in recent time, default to 0 which is uniform sampling')
    parser.add_argument('--pos_dim', type=int, default=172, help='dimension of the positional embedding')
    parser.add_argument('--pos_sample', type=str, default='binary', choices=['multinomial', 'binary'], help='two equivalent sampling method with empirically different running time')
    parser.add_argument('--he_encode_hid_dim', type=int, default=32, help='dimension of the hidden embedding of hypergraph encoder')
    parser.add_argument('--he_encode_out_dim', type=int, default=64, help='dimension of the hypergraph embeddding')
    parser.add_argument('--walk_encode_time_dim', type=int, default=32, help='dimension of the time embeddding in walk encoder')
    parser.add_argument('--walk_encode_hid_dim', type=int, default=32, help='dimension of the hidden embeddding in walk encoder')
    parser.add_argument('--walk_encode_out_dim', type=int, default=64, help='dimension of the walk embeddding')
    parser.add_argument('--src_he_encode_hid_dim', type=int, default=32, help='dimension of the hidden embedding of source hypergraph encoder')
    parser.add_argument('--src_he_encode_out_dim', type=int, default=64, help='dimension of the source hypergraph embeddding')
    parser.add_argument('--task_layer1_out_dim', type=int, default=64, help='dimension of the output of the first task layer')
    parser.add_argument('--max_he_size', type=int, default=25, help='maximum size (number of nodes) of a hypergraph')
    parser.add_argument('--walk_agg', type=str, default='set_node_gran', choices=['set_node_gran', 'mean_he_gran', 'mean_node_gran'], help='aggregation method of walk encoding to find source hyperedge encoding')
    parser.add_argument('--sampled_he_per_node', type=int, default=3, help='number of sampled he per node in node classification')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')  
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0, help='tolerated marginal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
    parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')

    # for time complexity experiment
    parser.add_argument('--partial_e_num', type=int, default=10000, help='number of hyperedges to load from the dataset(when loading the dataset partially)')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

def set_pretrained_parameters(data):
    if data == 'NDC-classes':
        return True, ['4'], 2, 'i', 2e-07, 'set_node_gran'
    if data == 'contact-high-school':
        return True, ['2'], 3, 'i', 2e-07, 'set_node_gran'
    if data =='congress-bills': #takes a while
        return True, ['2'], 3, 'i', 2e-7, 'mean_node_gran'
    if data =='tags-math-sx':
        return True, ['2'], 3, 'i', 2e-07, 'set_node_gran'

    return False, None, None, None, None, None