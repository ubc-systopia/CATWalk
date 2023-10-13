import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
from utils import *
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def train_val(train_val_data, model, mode, bs, epochs, criterion, optimizer, early_stopper, ngh_finders, he_infos, rand_samplers, logger):
    # unpack the data, prepare for the training
    train_he_info, val_he_info = train_val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
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
    
    num_he_instances = len(train_he_info)
    num_batch = math.ceil(num_he_instances / bs)
    logger.info('num of training instances: {}'.format(num_he_instances))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = list(train_he_info.keys())
    for epoch in range(epochs):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_he_instances - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, he_offset_l_cut, ts_l_cut = construct_algo_data_given_he_ids(batch_idx, train_he_info)
            size = len(batch_idx)
            #generate negative samples (fake hyperedges)
            src_l_fake = train_rand_sampler.sample(src_l_cut, he_offset_l_cut)

            # feed in the data and learn from error
            optimizer.zero_grad()
            model.train()
            pos_prob, neg_prob = model.contrast(src_l_cut, src_l_fake, he_offset_l_cut, ts_l_cut)   # the core training code
        
            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False).unsqueeze(1)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False).unsqueeze(1)

            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()

            # collect training results
            with torch.no_grad():
                model.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label.flatten() == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for {} nodes'.format(mode), model, val_rand_sampler, val_he_info)
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        # logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
        # logger.info('train f1: {}, val f1: {}'.format(np.mean(f1), val_f1))

        # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))


