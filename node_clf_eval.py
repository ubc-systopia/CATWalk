import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from utils import *
import torch.nn.functional as F


def nc_eval_one_epoch(hint, tgan, he_info, test_nodes, node_labels_mapping, sampled_he_per_node):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(test_nodes)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        nodes_list = list(test_nodes)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            batch_nodes= nodes_list[s_idx:e_idx]
            src_l_cut, he_offset_l_cut, ts_l_cut = construct_algo_data_given_nodes(batch_nodes, he_info, sampled_he_per_node)

            out = tgan.predict(src_l_cut, he_offset_l_cut, ts_l_cut, test=True)
            pred = F.log_softmax(out, dim=1)

            pred_label = pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
            true_label = np.array([node_labels_mapping[n]-1 for n in batch_nodes])
            pred_score = pred[:,1].detach().cpu().numpy()
            
            val_acc.append((pred_label == true_label).mean())
    return np.mean(val_acc)