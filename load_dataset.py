
def convert_strList_to_intList(l):
    return [int(x) for x in l]

def normalize_ts(ts, conv_rate = 1):
    min_t = min(ts)
    return [(t-min_t)/conv_rate for t in ts]

ts_normalization_coef = {
    'email-Enron': 10000, # (original time in milliseconds)
    'NDC-classes': 10000000000, # (original timestamp is in days) 
    'contact-high-school': 1, # (resolution of 20 seconds | recording all interactions from the previous 20 seconds)
    'contact-primary-school': 1, # (resolution of 20 seconds | recording all interactions from the previous 20 seconds)
    'congress-bills': 1, # (Timestamps are in days)
    'tags-math-sx': 1000000, 
    'email-Eu':10,
    'threads-ask-ubuntu':10000,
    
    'NDC-substances': 10000000,
    'tags-ask-ubuntu':1000000
}

def load_dataset(dataset):
    """
    Output:
        n_v: a list including the number of nodes per hyperedge
        v_simplices: a list of nodes of hyper edges that v_simplices[v_start_idx : v_start_idx + n_v[i]] are nodes of he_i
        ts: a list including the time stamps of hyperedges
    """
    path='HG_Data/' + dataset + '/'

    file_prefix = path + dataset

    n_v_path = file_prefix + "-nverts.txt"
    v_simplices_path = file_prefix + "-simplices.txt"
    ts_path = file_prefix + "-times.txt"

    with open(n_v_path) as n_v_file:
        n_v = n_v_file.read().splitlines()

    with open(v_simplices_path) as v_simplices_file:
        v_simplices = v_simplices_file.read().splitlines()

    with open(ts_path) as ts_file:
        ts = ts_file.read().splitlines()

    n_v = convert_strList_to_intList(n_v)
    v_simplices = convert_strList_to_intList(v_simplices)
    ts = convert_strList_to_intList(ts)

    ts = normalize_ts(ts, ts_normalization_coef[dataset])

    return n_v, v_simplices, ts, dataset

def n_classification_load_dataset(dataset):
    path='HG_Data/Node_Bin_Class_Detection_Data/' + dataset + '/'
    
    hes_path = path + "hyperedges-" + dataset + ".txt"
    node_labels_path = path + "node-labels-" + dataset + ".txt"
    label_names_path = path + "label-names-" + dataset + ".txt"

    with open(hes_path) as hes_file:
        hes = hes_file.read().splitlines()

    with open(node_labels_path) as node_labels_file:
        node_labels = node_labels_file.read().splitlines()

    with open(label_names_path) as label_names_file:
        label_names = label_names_file.read().splitlines()

    node_labels = convert_strList_to_intList(node_labels)
    
    return hes, node_labels, label_names
