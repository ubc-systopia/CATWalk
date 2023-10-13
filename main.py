from utils import *
from Hgraph import *
from load_dataset import *
import pickle
from module import *
import numpy as np
import random
from log import *
from train import *

args, sys_argv = get_args()

DATA = args.data

PRETRAINED = (args.pretrained == 'True') 
if PRETRAINED:
    found_params, nd, nl, m, b, wa = set_pretrained_parameters(DATA)
    if found_params:
        args.n_degree, args.n_layer, args.mode, args.bias, args.walk_agg = nd, nl, m, b, wa
    else:
        PRETRAINED = False
        
NUM_NEIGHBORS = args.n_degree
NUM_LAYER = args.n_layer
POS_DIM = args.pos_dim 
HE_ENCODE_HIDDEN_DIMS = args.he_encode_hid_dim 
HE_ENCODE_OUT_DIM = args.he_encode_out_dim 
SRC_HE_ENCODE_HIDDEN_DIMS = args.src_he_encode_hid_dim 
SRC_HE_ENCODE_OUT_DIM = args.src_he_encode_out_dim 
WALK_ENCODE_TIME_DIM = args.walk_encode_time_dim 
WALK_ENCODE_HIDDEN_DIM = args.walk_encode_hid_dim 
WALK_ENCODE_OUT_DIM = args.walk_encode_out_dim 
TASK_LAYER1_OUT_DIM = args.task_layer1_out_dim 
MAX_HE_SIZE = args.max_he_size 
WALK_AGG = args.walk_agg

NUM_EPOCH = args.n_epoch
BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
DROP_OUT = args.drop_out
TOLERANCE = args.tolerance

SEED = args.seed
GPU = args.gpu
CPU_CORES = args.cpu_cores
VERBOSITY = args.verbosity


assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path, runtime_id = set_up_logger(args, sys_argv)

### Load Data 
n_v, v_simplices, ts, dataset_name =  load_dataset(DATA)

### Generte basic hypergraph modelling (he_info)
full_he_info = generate_he_info(n_v, ts, v_simplices)
total_node_set = set(np.unique(np.array(v_simplices)))
num_total_unique_nodes = len(total_node_set)
num_total_hyperedges = len(n_v)

# split and pack the data by generating valid train/val/test mask according to the "mode"
ts_l = np.array(ts)
val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))
if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_he_ids = np.where(ts_l <= val_time)[0] + 1 
    valid_val_he_ids = np.where((ts_l > val_time) & (ts_l <= test_time))[0] + 1
    valid_test_he_ids = np.where(ts_l > test_time)[0] + 1

else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    hes_ids_after_val_time = np.where((ts_l > val_time))[0] + 1 
    he_nodes_after_val_time = set().union(*[full_he_info[i][0] for i in hes_ids_after_val_time]) 
    mask_node_set = set(random.sample(he_nodes_after_val_time, int(0.1 * num_total_unique_nodes))) 
    he_has_masked_nodes = np.array([len(full_he_info[i][0] & mask_node_set) > 0 for i in range(1, num_total_hyperedges+1)])

    valid_train_he_ids = np.where((ts_l <= val_time)  & ~(he_has_masked_nodes))[0]+1# Train edges can not contain any masked nodes
    valid_val_he_ids = np.where((ts_l > val_time) & (ts_l <= test_time) & ~(he_has_masked_nodes))[0]+1# Val edges can not contain any masked nodes
    valid_test_he_ids = np.where((ts_l > test_time) & (he_has_masked_nodes))[0]+1# test edges must contain at least one masked node
    
    he_is_all_masked_nodes = np.array([len(full_he_info[i][0] & mask_node_set) == min(len(full_he_info[i][0]), len(mask_node_set)) for i in range(1, num_total_hyperedges+1)])
    valid_test_all_new_he_ids = np.where((ts_l > test_time) & (he_is_all_masked_nodes))[0]+1
    valid_test_new_old_he_ids = np.setdiff1d(valid_test_he_ids, valid_test_all_new_he_ids)
    
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))
    logger.info('Out of {} test hyperedges, {} are all_new and {} are new_old'.format(len(valid_test_he_ids), len(valid_test_all_new_he_ids), len(valid_test_new_old_he_ids)))

# split data according to the mask
train_data = {key: full_he_info[key] for key in valid_train_he_ids}
val_data = {key: full_he_info[key] for key in valid_val_he_ids}
test_data = {key: full_he_info[key] for key in valid_test_he_ids}
if args.mode == 'i':
    test_all_new_data = {key: full_he_info[key] for key in valid_test_all_new_he_ids}
    test_new_old_data = {key: full_he_info[key] for key in valid_test_new_old_he_ids}
train_val_data = (train_data, val_data)

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one
max_node_idx = max(v_simplices)
assert(min(v_simplices) > 0)
full_n_adj_list = build_node_temporal_adjlist(max_node_idx, full_he_info)
full_ngh_finder = NeighborFinder(full_n_adj_list, full_he_info, bias=args.bias, sample_method=args.pos_sample)
#only including nodes and he in train and val
train_and_val_he_ids = np.union1d(valid_train_he_ids, valid_val_he_ids)
partial_he_info = {key: full_he_info[key] for key in train_and_val_he_ids}
nodes_partial_info = set().union(*[partial_he_info[i][0] for i in partial_he_info])
partial_max_node_idx = max(nodes_partial_info)
partial_adj_list = build_node_temporal_adjlist(partial_max_node_idx, partial_he_info)
partial_ngh_finder = NeighborFinder(partial_adj_list, partial_he_info, bias=args.bias, sample_method=args.pos_sample)
ngh_finders = partial_ngh_finder, full_ngh_finder
he_infos = partial_he_info, full_he_info

# create random samplers to generate train/val/test fake instances
train_nodes = set().union(*[train_data[i][0] for i in train_data])
val_nodes = set().union(*[val_data[i][0] for i in val_data])
test_nodes = set().union(*[test_data[i][0] for i in test_data])
train_rand_sampler = RandHyperEdgeSampler([train_nodes])
val_rand_sampler = RandHyperEdgeSampler([train_nodes, val_nodes])
test_rand_sampler = RandHyperEdgeSampler([train_nodes, val_nodes, test_nodes])
rand_samplers = train_rand_sampler, val_rand_sampler


# model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

catn = CATN(he_encoder_hidden_channels=HE_ENCODE_HIDDEN_DIMS, he_encoder_out_channels=HE_ENCODE_OUT_DIM,
            walk_encoder_time_channels=WALK_ENCODE_TIME_DIM, walk_encoder_hidden_channel=WALK_ENCODE_HIDDEN_DIM, walk_encoder_out_channels=WALK_ENCODE_OUT_DIM,
            src_he_encoder_hidden_channels=SRC_HE_ENCODE_HIDDEN_DIMS, src_he_encoder_out_channels=SRC_HE_ENCODE_OUT_DIM,
            task_layer1_out_size=TASK_LAYER1_OUT_DIM,
            num_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS, pos_dim=POS_DIM, max_he_size=MAX_HE_SIZE,
            verbosity=VERBOSITY, cpu_cores=CPU_CORES, get_checkpoint_path=get_checkpoint_path, 
            he_encoder_dropout=DROP_OUT, walk_encoder_dropout=DROP_OUT, src_he_encoder_dropout=DROP_OUT,
            walk_agg=WALK_AGG)

if PRETRAINED:
    logger.info('Lodaing pretrained model...')
    pretrained_model_path = "pretrained_models/"+ DATA +"/model.pth"
    catn.load_state_dict(torch.load(pretrained_model_path))

catn.to(device) 
optimizer = torch.optim.Adam(catn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

if not PRETRAINED:
    # start train and val phases
    train_val(train_val_data, catn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, he_infos, rand_samplers, logger)

# final testing
catn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
catn.update_he_info(full_he_info)
_, test_ap, _, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), catn, test_rand_sampler, test_data)
logger.info('Test statistics: {} all nodes -- auc: {}, ap: {}'.format(args.mode, test_auc, test_ap))
if args.mode == 'i':
    _, test_new_old_ap, _, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), catn, test_rand_sampler, test_new_old_data)
    logger.info('Test statistics: {} new_old nodes -- auc: {}, ap: {}'.format(args.mode, test_new_old_auc, test_new_old_ap))


# save model
logger.info('Saving CATN model ...')
torch.save(catn.state_dict(), best_model_path)
logger.info('CATN model saved')

# save one line result
save_oneline_result('log/', args, [test_auc, test_ap], runtime_id)

