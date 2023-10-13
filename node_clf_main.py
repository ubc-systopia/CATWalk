from utils import *
from Hgraph import *
from load_dataset import *
import pickle
from node_clf_module import *
import numpy as np
import random
from log import *
from node_clf_train import *

args, sys_argv = get_args()

DATA = args.data

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
SAMPLED_HE_PER_NODE = args.sampled_he_per_node

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
hes, node_labels, label_names = n_classification_load_dataset(DATA)

num_nodes = len(node_labels)
num_second_label = np.sum(np.array(node_labels) ==2)

### Generte basic hypergraph modelling (he_info)
full_he_info = generate_he_info(n_v, ts, v_simplices)
hes_list, node_labels_mapping, label_name_mapping = generate_nc_data_structures(hes, node_labels, label_names)

all_node_labels = set(np.unique(np.array(node_labels)))
num_node_labels = len(all_node_labels)
total_node_set = set(np.unique(np.array(v_simplices)))
num_total_unique_nodes = len(total_node_set)
num_total_hyperedges = len(n_v)

# split and pack the data by generating valid train/val/test mask according to the "mode"
test_nodes = set(random.sample(total_node_set, int(0.2 * num_total_unique_nodes))) 
remained_nodes = total_node_set - test_nodes
val_nodes = set(random.sample(remained_nodes, int(0.1 * num_total_unique_nodes))) 
train_nodes = remained_nodes - val_nodes
mask_node_set = test_nodes

logger.info('Sampled {} nodes (20 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))
logger.info('Training: on {} nodes and sampling {} hyperedges per node in training'.format(len(train_nodes), SAMPLED_HE_PER_NODE))
logger.info('Node_count: Training nodes: {} nodes | Validation nodes: {} | Test nodes: {}'.format(len(train_nodes), len(val_nodes), len(test_nodes)))

he_has_masked_nodes = np.array([len(full_he_info[i][0] & mask_node_set) > 0 for i in range(1, num_total_hyperedges+1)])
if args.mode == 't':
    logger.info('Transductive training...')

    train_data = full_he_info
    test_data = full_he_info
else:
    assert(args.mode == 'i')
    valid_train_he_ids = np.where(~(he_has_masked_nodes))[0]+1 # train edges can not contain any masked nodes
    valid_test_he_ids = np.where(he_has_masked_nodes)[0]+1 # test edges must contain at least one masked node

    train_data = {key: full_he_info[key] for key in valid_train_he_ids}
    test_data = full_he_info

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train phases use the partial one
# while test phase still always uses the full one
max_node_idx = max(v_simplices)
assert(min(v_simplices) > 0)
full_n_adj_list = build_node_temporal_adjlist(max_node_idx, full_he_info)
full_ngh_finder = NeighborFinder(full_n_adj_list, full_he_info, bias=args.bias, sample_method=args.pos_sample)
# only including nodes and he in train and val

partial_he_info = train_data
nodes_partial_info = remained_nodes
partial_max_node_idx = max(nodes_partial_info)
partial_adj_list = build_node_temporal_adjlist(partial_max_node_idx, partial_he_info)
partial_ngh_finder = NeighborFinder(partial_adj_list, partial_he_info, bias=args.bias, sample_method=args.pos_sample)

ngh_finders = partial_ngh_finder, full_ngh_finder
he_infos = partial_he_info, full_he_info


# model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

catn = nc_CATN(he_encoder_hidden_channels=HE_ENCODE_HIDDEN_DIMS, he_encoder_out_channels=HE_ENCODE_OUT_DIM,
            walk_encoder_time_channels=WALK_ENCODE_TIME_DIM, walk_encoder_hidden_channel=WALK_ENCODE_HIDDEN_DIM, walk_encoder_out_channels=WALK_ENCODE_OUT_DIM,
            src_he_encoder_hidden_channels=SRC_HE_ENCODE_HIDDEN_DIMS, src_he_encoder_out_channels=SRC_HE_ENCODE_OUT_DIM,
            task_layer1_out_size=TASK_LAYER1_OUT_DIM,
            num_node_classes= num_node_labels,
            sampled_he_per_node=SAMPLED_HE_PER_NODE,
            num_layers=NUM_LAYER, num_neighbors=NUM_NEIGHBORS, pos_dim=POS_DIM, max_he_size=MAX_HE_SIZE,
            verbosity=VERBOSITY, cpu_cores=CPU_CORES, get_checkpoint_path=get_checkpoint_path, 
            he_encoder_dropout=DROP_OUT, walk_encoder_dropout=DROP_OUT, src_he_encoder_dropout=DROP_OUT,
            walk_agg=WALK_AGG)

catn.to(device) 
optimizer = torch.optim.Adam(catn.parameters(), lr=0.01)

# handle unbalanced data
num_first_label = num_nodes - num_second_label
criterion = torch.nn.NLLLoss(weight=torch.Tensor([num_second_label/(2*num_nodes) , num_first_label/(2*num_nodes)]).to(device))

early_stopper = EarlyStopMonitor(tolerance=0, max_round=10)

# start train and val phases
nc_train_val(train_data, train_nodes, val_nodes, node_labels_mapping, catn, args.mode, SAMPLED_HE_PER_NODE, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, ngh_finders, he_infos, early_stopper, logger)

# final testing
catn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
catn.update_he_info(full_he_info)
test_acc = nc_eval_one_epoch('test for {} nodes'.format(args.mode), catn, test_data, test_nodes, node_labels_mapping, SAMPLED_HE_PER_NODE)
logger.info('Test statistics: {} all nodes -- acc: {}'.format(args.mode, test_acc))

# save model
logger.info('Saving CATN model ...')
torch.save(catn.state_dict(), best_model_path)
logger.info('CATN model saved')

# save one line result
save_oneline_result('log/', args, [test_acc], runtime_id)

