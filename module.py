import logging
import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from position import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F

from mlp_mixer import *
from set_mixer import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class CATN(torch.nn.Module):
    def __init__(self,  
                he_encoder_hidden_channels, he_encoder_out_channels,
                walk_encoder_time_channels, walk_encoder_hidden_channel, walk_encoder_out_channels,
                src_he_encoder_hidden_channels, src_he_encoder_out_channels,
                task_layer1_out_size,
                
                num_layers=3,  num_neighbors=20, pos_dim=0, cpu_cores=1, 
                verbosity=1,
                max_he_size=25,

                he_encoder_num_layers=2, he_encoder_dropout=0.5, 
                he_encoder_channel_expansion_factor=4, he_encoder_module_spec=None, he_encoder_use_single_layer=False,
                
                walk_encoder_num_layers=2, walk_encoder_dropout=0.5, walk_encoder_token_expansion_factor=0.5, 
                walk_encoder_channel_expansion_factor=4, walk_encoder_module_spec=None, walk_encoder_use_single_layer=False,

                src_he_encoder_num_layers=2, src_he_encoder_dropout=0.5, 
                src_he_encoder_channel_expansion_factor=4, src_he_encoder_module_spec=None, src_he_encoder_use_single_layer=False,
                
                get_checkpoint_path=None, 
                walk_agg="set_nodeـgran"               
                ):
        super(CATN, self).__init__()
        self.logger = logging.getLogger(__name__)

        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, num_layers)
        self.ngh_finder = None
        self.max_he_size = max_he_size

        self.pos_dim = pos_dim  # position feature dimension
        self.logger.info('neighbors: {}, pos dim: {}'.format(self.num_neighbors, self.pos_dim))

        # hyperedge information
        self.he_info = None

        self.walk_encoder_out_channels = walk_encoder_out_channels

        # embedding layers and encoders
        self.position_encoder = PositionEncoder(num_layers=self.num_layers, max_he_size=max_he_size, enc_dim=self.pos_dim, he_info= self.he_info,
                                                ngh_finder=self.ngh_finder, verbosity=verbosity, cpu_cores=cpu_cores, logger=self.logger)
        self.edge_pos_encoder = SetMixer(per_graph_size=self.max_he_size, input_channels=self.max_he_size*(self.num_layers+1) , 
                                        hidden_channels=he_encoder_hidden_channels , out_channels=he_encoder_out_channels,
                                        num_layers=he_encoder_num_layers, dropout=he_encoder_dropout, channel_expansion_factor=he_encoder_channel_expansion_factor, 
                                        module_spec=he_encoder_module_spec, use_single_layer=he_encoder_use_single_layer)
        self.walk_encoder = MLPMixer(per_graph_size=self.num_layers, time_channels=walk_encoder_time_channels,
                                        input_channels=he_encoder_out_channels, hidden_channels=walk_encoder_hidden_channel, 
                                        out_channels=walk_encoder_out_channels,
                                        num_layers=walk_encoder_num_layers, dropout=walk_encoder_dropout,
                                        token_expansion_factor=walk_encoder_token_expansion_factor, channel_expansion_factor=walk_encoder_channel_expansion_factor, 
                                        module_spec=walk_encoder_module_spec, use_single_layer=walk_encoder_use_single_layer)
        self.src_edge_encoder = SetMixer(per_graph_size=self.max_he_size, input_channels=self.walk_encoder_out_channels , 
                                        hidden_channels=src_he_encoder_hidden_channels , out_channels=src_he_encoder_out_channels,
                                        num_layers=src_he_encoder_num_layers, dropout=src_he_encoder_dropout, channel_expansion_factor=src_he_encoder_channel_expansion_factor, 
                                        module_spec=src_he_encoder_module_spec, use_single_layer=src_he_encoder_use_single_layer)

        # final projection layer
        self.walk_agg = walk_agg 
        if(walk_agg == "mean_he_gran"):
            self.task_output_fc1 = torch.nn.Linear(walk_encoder_out_channels, task_layer1_out_size) 
        elif(self.walk_agg == "mean_node_gran"):
            self.task_output_fc1 = torch.nn.Linear(walk_encoder_out_channels * max_he_size, task_layer1_out_size) 
        else:#set_node_gran
            self.task_output_fc1 = torch.nn.Linear(src_he_encoder_out_channels, task_layer1_out_size) 
       
        self.task_output_act = torch.nn.ReLU()
        self.task_output_fc2 = torch.nn.Linear(task_layer1_out_size, 1) 

        self.get_checkpoint_path = get_checkpoint_path

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.position_encoder.ngh_finder = ngh_finder
    
    def update_he_info(self, he_info):
        he_info[0] = (set([0]), 0) #padding he and node (used when no neighbors are available)
        self.he_info = he_info
        self.position_encoder.he_info = he_info
        
    def grab_subgraph(self, src_idx_l, cut_time_l):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors)
        return subgraph

    def contrast(self, pos_src_idx_l, neg_src_idx_l, he_offset_l, cut_time_l, test=False):
        """
        # he_offset_l: showing which nodes are in the same potential hyperedge (len = #he + 1)

        1. grab subgraph for pos src and neg src
        2. forward propagate to get pos src embeddings (and finally pos_score (shape: [batch, ]))
        3. forward propagate to get neg src embeddings (and finally neg_score (shape: [batch, ]))
        """
        start = time.time()
        subgraph_pos_src_idx_l = self.grab_subgraph(pos_src_idx_l, cut_time_l)
        subgraph_neg_src_idx_l = self.grab_subgraph(neg_src_idx_l, cut_time_l)
        end = time.time()

        pos_score = self.forward(pos_src_idx_l, he_offset_l, cut_time_l, subgraph_pos_src_idx_l, test=test)

        neg_score = self.forward(neg_src_idx_l, he_offset_l, cut_time_l, subgraph_neg_src_idx_l, test=test)
        
        return pos_score.sigmoid(), neg_score.sigmoid()

    def forward(self, src_idx_l, he_offset_l, cut_time_l, subgraph_src, test=False, nwalks_per_batch=16):

        self.position_encoder.init_internal_data(src_idx_l, cut_time_l, subgraph_src)
        self.position_encoder.node_pos_encoding(src_idx_l, he_offset_l)
        num_source_he, he_n_walks_l, walk_he_emb_matrix, walk_src_neighbors_ts, num_walks_per_src_node = self.position_encoder.hyperedge_pos_encoding_prepare(he_offset_l, src_idx_l, subgraph_src)
        
        n_hop = self.num_layers
        edge_encoder_batch_size = nwalks_per_batch*n_hop
        n_walks = len(walk_he_emb_matrix)

        x = torch.Tensor(np.array(walk_he_emb_matrix))
        x = torch.split(x, edge_encoder_batch_size)
        
        encoded_hes = torch.Tensor().to(device)
        for batch_data in x:
            batch_data = batch_data.to(device)
            encoded_he_batch = self.edge_pos_encoder(batch_data)
            encoded_hes = torch.cat((encoded_hes, encoded_he_batch), 0)
        
        encoded_hes_walks= torch.split(encoded_hes, n_hop)
        encoded_hes_walks = torch.stack(encoded_hes_walks)
        
        he_ts_walks = torch.Tensor(walk_src_neighbors_ts.reshape((walk_src_neighbors_ts.shape[0]*walk_src_neighbors_ts.shape[1]), walk_src_neighbors_ts.shape[2])).to(device)
        
        encoded_walks = self.walk_encoder(encoded_hes_walks, he_ts_walks, batch_size=np.sum(np.array(he_n_walks_l)))
       

        def take_mean(x):
                return torch.mean(x, dim=0)

        if(self.walk_agg == "mean_he_gran"):
            encoded_walks= torch.split(encoded_walks, he_n_walks_l)
            
            encoded_src_hes = torch.stack(list(map(take_mean, encoded_walks)))
       
        elif(self.walk_agg == "mean_node_gran"):
            encoded_walks= torch.split(encoded_walks, num_walks_per_src_node)
            encoded_walks = torch.stack(encoded_walks)
            encoded_src_nodes = torch.mean(encoded_walks, dim=1)
           
            he_n_nodes = [he_offset_l[idx+1]-he_offset_l[idx] for idx in range(num_source_he)]
            encoded_src_hes = torch.split(encoded_src_nodes, he_n_nodes)

            def zero_pad(encoded_nodes_src_he):
                encoded_nodes_src_he = encoded_nodes_src_he.flatten()
                out = torch.zeros(self.walk_encoder_out_channels * self.max_he_size).to(device)
                out[:len(encoded_nodes_src_he)] = encoded_nodes_src_he
                return out

            encoded_src_hes = torch.stack(list(map(zero_pad, encoded_src_hes)))
       
        else:# "set_node_gran"
            encoded_walks= torch.split(encoded_walks, num_walks_per_src_node)
            encoded_walks = torch.stack(encoded_walks)
            encoded_src_nodes = torch.mean(encoded_walks, dim=1)

            he_n_nodes = [he_offset_l[idx+1]-he_offset_l[idx] for idx in range(num_source_he)]
            pre_encoded_src_hes = torch.split(encoded_src_nodes, he_n_nodes)

            def zero_pad_2d(encoded_nodes_src_he):
                out = torch.zeros(self.max_he_size, self.walk_encoder_out_channels).to(device)
                out[:encoded_nodes_src_he.shape[0], :] = encoded_nodes_src_he
                return out
            pre_encoded_src_hes = torch.stack(list(map(zero_pad_2d, pre_encoded_src_hes)))            
            encoded_src_hes = self.src_edge_encoder(pre_encoded_src_hes)

        h = self.task_output_act(self.task_output_fc1(encoded_src_hes))
        score = self.task_output_fc2(h)

        return score
 

class PositionEncoder(nn.Module):
    def __init__(self, num_layers, max_he_size=25, enc_dim=2, he_info=None, ngh_finder=None, verbosity=1, cpu_cores=1, logger=None):
        super(PositionEncoder, self).__init__()

        self.num_layers = num_layers#number of hops
        self.max_he_size = max_he_size
        self.enc_dim = enc_dim
        self.ngh_finder = ngh_finder
        self.he_info = he_info
        self.verbosity = verbosity
        self.cpu_cores = cpu_cores
        self.logger = logger

        self.node2emb_maps = None # mapping from a visited node to positional vector in walks starting from a src node
        self.visited_nodes = None # mapping from index of src node in src_idx_l to set of nodes visited by its subgraph(setwalks)
        self.node2posemb = None # mapping from a visited node to positional embedding in walks starting from a src hyperedge
        
    def init_internal_data(self, src_idx_l, cut_time_l, subgraph_src):

        if self.enc_dim == 0:
            return
        start = time.time()
        # initialize internal data structure to index node positions
        self.node2emb_maps, self.visited_nodes = self.collect_pos_mapping_ptree(src_idx_l, cut_time_l, subgraph_src)
        
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('init positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    
    def collect_pos_mapping_ptree(self, src_idx_l, cut_time_l, subgraph_src):
        # Input:
        # src_idx_l: list of nodes starting from them
        # subgraph_src: subgraphs from nodes in src_idx_l (a series of hyperedges)
        # Return:
        # node2emb_maps: a list of dict {(batch-node index) -> embedding of node(of size h_hop+1)}


        if self.cpu_cores == 1:
            _, subgraph_src_he, subgraph_src_ts = subgraph_src  
            node2emb_maps = {}
            visited_nodes = {}
            for row in range(len(src_idx_l)):
                src = src_idx_l[row]
                cut_time = cut_time_l[row]
                src_neighbors_he = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_he]
                src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
                node2emb_map, visited_n = self.collect_pos_mapping_ptree_sample(src, cut_time,
                                                                   src_neighbors_he, src_neighbors_ts, batch_idx=row)
                node2emb_maps.update(node2emb_map)
                visited_nodes.update(visited_n)           

        else:
            # multiprocessing version, no significant gain though
            cores = self.cpu_cores
            if cores in [-1, 0]:
                cores = mp.cpu_count()
            pool = mp.Pool(processes=cores)
            node2emb_maps, visited_nodes = pool.map(self.collect_pos_mapping_ptree_sample_mp,
                                         [(src_idx_l, cut_time_l, subgraph_src, row) for row in range(len(src_idx_l))],
                                         chunksize=len(src_idx_l)//cores+1)
            pool.close()
        
        return node2emb_maps, visited_nodes

    def collect_pos_mapping_ptree_sample(self, src, cut_time, src_neighbors_he, src_neighbors_ts,
                                         batch_idx):
        
        n_hop = self.num_layers
        makekey = entity2key
        node2emb = {}
        visited_n = {}
        visited_ngh_nodes = set()
       
        # landing probability encoding, n_hop+1 types of probabilities for each node
        
        # src node
        visited_ngh_nodes.update([src])
        src_node_key = makekey(batch_idx, src)
        node2emb[src_node_key] = np.zeros(n_hop+1, dtype=np.float32)
        node2emb[src_node_key][0] = 1
        #visited nodes in the set walk
        for k in range(n_hop):
            k_hop_total = len(src_neighbors_he[k])
           
            for ngh_he, ngh_ts in zip(src_neighbors_he[k], src_neighbors_ts[k]):
                ngh_he_nodes = self.he_info[ngh_he][0]
                visited_ngh_nodes.update(ngh_he_nodes)

                for node in ngh_he_nodes:
                    ngh_node_key = makekey(batch_idx, node)
                    if ngh_node_key not in node2emb:
                        node2emb[ngh_node_key] = np.zeros(n_hop+1, dtype=np.float32)
                    node2emb[ngh_node_key][k+1] += 1/k_hop_total # convert into landing probabilities by normalizing with k hop sampling number
        null_key = makekey(batch_idx, 0)
        node2emb[null_key] = np.zeros(n_hop+1, dtype=np.float32)
        visited_n[batch_idx] = list(visited_ngh_nodes)
        return node2emb, visited_n
    
    def collect_pos_mapping_ptree_sample_mp(self, args):
        src_idx_l, cut_time_l, subgraph_src, row, enc = args

        _, subgraph_src_he, subgraph_src_ts = subgraph_src
        src = src_idx_l[row]
        cut_time = cut_time_l[row]
        
        src_neighbors_he = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_he]
        src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
        node2emb_map, visited_nodes = self.collect_pos_mapping_ptree_sample(src, cut_time,
                                                            src_neighbors_he, src_neighbors_ts, batch_idx=row)
        
        return node2emb_map, visited_nodes

    def node_pos_encoding(self, src_idx_l, he_offset_l):
        """
        Generate positional embedding for visited nodes
        he_offset_l: the ranges for each hypergraph nodes came in src_idx_l (len == #he + 1)

        output{"src_he_idx - node_idx" : node_pos_emb (max_he_size * (n_hop+1))}
        """
        node2posemb = {}
        makekey = entity2key
        
        for he_idx in range(len(he_offset_l)-1):
            he_nodes_start = he_offset_l[he_idx]
            he_nodes_end = he_offset_l[he_idx+1]

            he_visited_nodes = set()
            for src_idx in range(he_nodes_start, he_nodes_end):
                he_visited_nodes.update(self.visited_nodes[src_idx])

            n_hop = self.num_layers
            max_he_size = self.max_he_size
            for v_node in he_visited_nodes:
                node_embedding = np.zeros((max_he_size, n_hop+1), dtype=np.float32)
                for idx, row in enumerate(range(he_nodes_start, he_nodes_end)):
                    node_key = makekey(row, v_node)
                    
                    pos_vector = None
                    if node_key in self.node2emb_maps:
                        pos_vector = self.node2emb_maps[node_key]
                    else:
                        pos_vector =  self.node2emb_maps[makekey(row, 0)]

                    node_embedding[idx] = pos_vector
                    
                emb_node_key = makekey(he_idx, v_node)
                node2posemb[emb_node_key] = node_embedding.flatten()
        self.node2posemb = node2posemb
    
    def subgraph_tree2walk(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def hyperedge_pos_encoding_prepare(self, he_offset_l, src_idx_l, subgraph_src):
        """
        build raw pos encoding for each visited hyperedge from the pos encoding of nodes
        he_offset_l: the ranges for each hypergraph nodes came in src_idx_l (len == #he + 1)

        outputs a matrix of encoding of hyperedges put in order of walks
        """
        getkey = entity2key
        _, subgraph_src_he, subgraph_src_ts = subgraph_src  
        n_hop = self.num_layers

        num_source_he = len(he_offset_l)-1

        num_walks_per_src_node = subgraph_src_he[-1].shape[-1]
        
        walk_src_neighbors_he = self.subgraph_tree2walk(subgraph_src_he)
        walk_src_neighbors_ts = self.subgraph_tree2walk(subgraph_src_ts)

        walk_he_emb_matrix = []

        he_n_walks_l = []
        for source_he_idx in range(num_source_he):
            he_n_walks = 0
            he_nodes_start, he_nodes_end = he_offset_l[source_he_idx], he_offset_l[source_he_idx+1]
            # we need to consider all walks from all nodes in source_he_idx hyperedge
            for row in range(he_nodes_start, he_nodes_end):#src nodes of this he
                walks_from_src_node = walk_src_neighbors_he[row]
                #iterate over all visited hyperedges in the walks and generate pos embedding for them
                for walk in walks_from_src_node:
                    he_n_walks += 1
                    for he in walk:
                        he_enc = self.get_he_pos_embedding(source_he_idx, he, n_hop)
                        walk_he_emb_matrix.append(he_enc) 
            he_n_walks_l.append(he_n_walks)

        walk_he_emb_matrix = np.array(walk_he_emb_matrix)
        
       
        return num_source_he, he_n_walks_l, walk_he_emb_matrix, walk_src_neighbors_ts, num_walks_per_src_node
            
    def get_he_pos_embedding(self, source_he_idx, visited_he, n_hop):
        he_nodes = self.he_info[visited_he][0]
        makekey = entity2key
        max_he_size = self.max_he_size
        
        he_embedding = np.zeros((max_he_size, max_he_size*(n_hop+1)), dtype=np.float32)
        for i, node in enumerate(he_nodes):
            he_embedding[i] = self.node2posemb[makekey(source_he_idx, node)]
        
        return he_embedding

    def get_walk_he_emb(self, batch_idx, batch_size):
        n_hop = self.num_layers
        
        return self.walk_he_emb_matrix[batch_idx*(batch_size*n_hop) : (batch_idx+1)*(batch_size*n_hop)]
