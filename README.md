# CAT-WALK: Inductive Hypergraph Learning via Set Walks

This repository is the official implementation of "CAT-WALK: Inductive Hypergraph Learning via Set Walks". 

## Requirements
* `python >= 3.7`, `PyTorch >= 1.4`, please refer to their official websites for installation details.
* Other dependencies:
```{bash}
pandas==0.24.2
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
numba==0.51.2
```

### Usage Summary
```txt
usage: Interface for CATWALK: Inductive Dynamic Representation Learning for Link Prediction on Temporal Hyper Graphs
       [-h]
       [-d {NDC-classes,contact-primary-school,contact-high-school,email-Enron,email-Eu,congress-bills,tags-math-sx,threads-ask-ubuntu,NDC-substances,tags-ask-ubuntu}]
       [-m {t,i}] [--pretrained {True,False}] [--n_degree [N_DEGREE [N_DEGREE ...]]] [--n_layer N_LAYER]
       [--bias BIAS] [--pos_dim POS_DIM] [--pos_sample {multinomial,binary}]
       [--he_encode_hid_dim HE_ENCODE_HID_DIM] [--he_encode_out_dim HE_ENCODE_OUT_DIM]
       [--walk_encode_time_dim WALK_ENCODE_TIME_DIM] [--walk_encode_hid_dim WALK_ENCODE_HID_DIM]
       [--walk_encode_out_dim WALK_ENCODE_OUT_DIM] [--src_he_encode_hid_dim SRC_HE_ENCODE_HID_DIM]
       [--src_he_encode_out_dim SRC_HE_ENCODE_OUT_DIM] [--task_layer1_out_dim TASK_LAYER1_OUT_DIM]
       [--max_he_size MAX_HE_SIZE] [--walk_agg {set_node_gran,mean_he_gran,mean_node_gran}]
       [--sampled_he_per_node SAMPLED_HE_PER_NODE] [--n_epoch N_EPOCH] [--bs BS] [--lr LR]
       [--drop_out DROP_OUT] [--tolerance TOLERANCE] [--seed SEED] [--gpu GPU] [--cpu_cores CPU_CORES]
       [--verbosity VERBOSITY] [--partial_e_num PARTIAL_E_NUM]
```

### Optional arguments
```{txt}
optional arguments:
  -h, --help            show this help message and exit
  -d {NDC-classes,contact-primary-school,contact-high-school,email-Enron,email-Eu,congress-bills,tags-math-sx,threads-ask-ubuntu,NDC-substances,tags-ask-ubuntu}, --data {NDC-classes,contact-primary-school,contact-high-school,email-Enron,email-Eu,congress-bills,tags-math-sx,threads-ask-ubuntu,NDC-substances,tags-ask-ubuntu}
                        data sources to use, try email-Enron or NDC-classes
  -m {t,i}, --mode {t,i}
                        transductive (t) or inductive (i)
  --pretrained {True,False}
                        use pretrained models or not
  --n_degree [N_DEGREE [N_DEGREE ...]]
                        a list of neighbor sampling numbers for different hops, when only a single element is
                        input n_layer will be activated
  --n_layer N_LAYER     number of network layers
  --bias BIAS           the hyperparameter alpha controlling sampling preference in recent time, default to 0
                        which is uniform sampling
  --pos_dim POS_DIM     dimension of the positional embedding
  --pos_sample {multinomial,binary}
                        two equivalent sampling method with empirically different running time
  --he_encode_hid_dim HE_ENCODE_HID_DIM
                        dimension of the hidden embedding of hypergraph encoder
  --he_encode_out_dim HE_ENCODE_OUT_DIM
                        dimension of the hypergraph embeddding
  --walk_encode_time_dim WALK_ENCODE_TIME_DIM
                        dimension of the time embeddding in walk encoder
  --walk_encode_hid_dim WALK_ENCODE_HID_DIM
                        dimension of the hidden embeddding in walk encoder
  --walk_encode_out_dim WALK_ENCODE_OUT_DIM
                        dimension of the walk embeddding
  --src_he_encode_hid_dim SRC_HE_ENCODE_HID_DIM
                        dimension of the hidden embedding of source hypergraph encoder
  --src_he_encode_out_dim SRC_HE_ENCODE_OUT_DIM
                        dimension of the source hypergraph embeddding
  --task_layer1_out_dim TASK_LAYER1_OUT_DIM
                        dimension of the output of the first task layer
  --max_he_size MAX_HE_SIZE
                        maximum size (number of nodes) of a hypergraph
  --walk_agg {set_node_gran,mean_he_gran,mean_node_gran}
                        aggregation method of walk encoding to find source hyperedge encoding
  --sampled_he_per_node SAMPLED_HE_PER_NODE
                        number of sampled he per node in node classification
  --n_epoch N_EPOCH     number of epochs
  --bs BS               batch_size
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability for all dropout layers
  --tolerance TOLERANCE
                        tolerated marginal improvement for early stopper
  --seed SEED           random seed for all randomized algorithms
  --gpu GPU             which gpu to use
  --cpu_cores CPU_CORES
                        number of cpu_cores used for position encoding
  --verbosity VERBOSITY
                        verbosity of the program output
  --partial_e_num PARTIAL_E_NUM
                        number of hyperedges to load from the dataset(when loading the dataset partially)
```
