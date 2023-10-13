import logging
import time
import sys
import os
from utils import *


def set_up_logger(args, sys_argv):
    # set up running log
    n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    n_degree = [str(n) for n in n_degree]
    runtime_id = '{}-{}-{}-{}-{}-{}'.format(str(time.time()), args.data, args.mode[0], n_layer, 'k'.join(n_degree), args.pos_dim)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_path = 'log/{}.log'.format(runtime_id)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    # set up model parameters log
    checkpoint_root = './saved_checkpoints/'
    checkpoint_dir = checkpoint_root + runtime_id + '/'
    best_model_root = './best_models/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.mkdir(checkpoint_root)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.mkdir(best_model_root)
        logger.info('Create directory'.format(best_model_root))
    os.mkdir(checkpoint_dir)
    os.mkdir(best_model_dir)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
    best_model_path = best_model_dir + 'best-model.pth'

    return logger, get_checkpoint_path, best_model_path, runtime_id


def save_oneline_result(dir, args, test_results, runtime_id):
    n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    n_degree = [str(n) for n in n_degree]
    with open(dir+'oneline_results.txt', 'a') as f:
        elements = [str(e) for e in [args.data, args.mode[0], n_layer, 'k'.join(n_degree), args.bias, args.pos_dim, *[str(v)[:6] for v in test_results], runtime_id]]
        f.write('\t'.join(elements)+'\n')
