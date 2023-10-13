import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from node_clf_eval import *
import logging
import random
from utils import *
import torch.nn.functional as F
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def nc_train_val(train_data, train_nodes, val_nodes, node_labels_mapping, model, mode, sampled_he_per_node, bs, epochs, criterion, optimizer, ngh_finders, he_infos, early_stopper, logger):
    # unpack the data, prepare for the training
    train_he_info = train_data
    partial_ngh_finder, full_ngh_finder = ngh_finders
    partial_he_info, full_he_info = he_infos
    if mode == 't':  # transductive
        model.update_ngh_finder(full_ngh_finder)
        model.update_he_info(full_he_info)
    elif mode == 'i':  # inductive
        model.update_ngh_finder(partial_ngh_finder)
        model.update_he_info(partial_he_info)
    else:
        raise ValueError('training mode {} not found.'.format(mode))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    
    num_instances = len(train_nodes)
    num_batch = math.ceil(num_instances / bs)
    logger.info('num of training instances: {}'.format(num_instances))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    nodes_list = list(train_nodes)
    for epoch in range(epochs):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(nodes_list)  # shuffle the training samples for every epoch
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_instances - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_nodes= nodes_list[s_idx:e_idx]
            src_l_cut, he_offset_l_cut, ts_l_cut = construct_algo_data_given_nodes(batch_nodes, train_he_info, sampled_he_per_node)
            
            # feed in the data and learn from error
            optimizer.zero_grad()
            model.train()

            out = model.predict(src_l_cut, he_offset_l_cut, ts_l_cut)# the core training code
            pred = F.log_softmax(out, dim=1)
 
            true = torch.Tensor([node_labels_mapping[n]-1 for n in batch_nodes]).type(torch.LongTensor).to(device)

            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            # collect training results
            with torch.no_grad():
                model.eval()

                true_label = true.detach().cpu().numpy()
                pred_label = pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
                pred_score = pred[:,1].detach().cpu().numpy()
                
                correct = true_label == pred_label
                acc.append(float(np.sum(correct))/len(correct))
 
                m_loss.append(loss.item())

        # validation phase use all information
        val_acc = nc_eval_one_epoch('val for {} nodes'.format(mode), model, train_he_info, val_nodes, node_labels_mapping, sampled_he_per_node)
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
 
        # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_acc):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
           torch.save(model.state_dict(), model.get_checkpoint_path(epoch))

