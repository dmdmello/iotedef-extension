#written by Zilin Shen and Daniel de Mello
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from models.lstm import Lstm
from seq2seq.utils import softmax, print_header, get_events
from seq2seq.seq2seq_attention import Seq2seqAttention
import argparse
import copy
import tensorflow as tf
import random
import os

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def get_args(jupyter_args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--permute_truncated', required=False, action='store_true', 
                        help="Bool for activating permutation invariance")
    parser.add_argument('--freeze_seq2seq', required=False, action='store_true', 
                        help="Bool for keeping seq2seq weights at each relabel round ")
    parser.add_argument('--use_prob_embedding', required=False, action='store_true', 
                        help="Bool for using original probability based embedding proposed in the original paper")
    parser.add_argument('--sequence_length', required=False, type=int, default=10, 
                        help="Length of truncated subsequences used in the seq2seq training")
    parser.add_argument('--rv', required=False, type=int, default=1, 
                        help="'round value' hyperparameter used for probability embedding, if activated")
    parser.add_argument('--ps_epochs', required=False, type=int, default=50, 
                        help="number of training epochs for per-step detectors")
    parser.add_argument('--s2s_epochs', required=False, type=int, default=30, 
                        help="number of training epochs for seq2seq detector")
    parser.add_argument('--relabel_rounds', required=False, type=int, default=1, 
                        help="Number of relabel rounds")
    parser.add_argument('--patience', required=False, type=int, default=None,
                        help="Patience for early stopping. Any value activates early stopping.")
    parser.add_argument('--data_fraction', required=False, type=float, default=1.0,
                        help="Fraction (between 0 and 1) for how much of the training set to use.")
    #parser.add_argument('--name', required=False, type=str, default=None,
    #                    help="Experiment name.")
    parser.add_argument('--use_wandb', required=False, action='store_true', 
                        help="Turn on logs with wandb")

    if jupyter_args is not None:
        args = parser.parse_args(jupyter_args)
    else: 
        args = parser.parse_args()
    return args

#jupyter_args = ['--permute_truncated', '--use_prob_embedding']
args = get_args()
args.data_fraction = min(args.data_fraction, 1)
seq2seq_config = {"sequence_length": args.sequence_length, 
                  "permute_truncated": args.permute_truncated,
                  "use_prob_embedding": args.use_prob_embedding,
                  "rv": args.rv
                  }   

# -----------get the preprocessed training and testing saved as .npy files
test_label_infection = np.load('preprocess/saved/test_label_infection.npy')
train_label_infection = np.load('preprocess/saved/train_label_infection.npy')
test_data_infection = np.load('preprocess/saved/test_data_infection.npy')
train_data_infection = np.load('preprocess/saved/train_data_infection.npy')

test_label_reconnaissance = np.load('preprocess/saved/test_label_reconnaissance.npy')
train_label_reconnaissance = np.load('preprocess/saved/train_label_reconnaissance.npy')
test_data_reconnaissance = np.load('preprocess/saved/test_data_reconnaissance.npy')
train_data_reconnaissance = np.load('preprocess/saved/train_data_reconnaissance.npy')

test_label_attack = np.load('preprocess/saved/test_label_attack.npy')
train_label_attack = np.load('preprocess/saved/train_label_attack.npy')
test_data_attack = np.load('preprocess/saved/test_data_attack.npy')
train_data_attack = np.load('preprocess/saved/train_data_attack.npy')

all_data = {"infection": 
                        {
                        'train': [train_data_infection, train_label_infection], 
                        'test': [test_data_infection, test_label_infection]
                        },
            "attack": 
                        {
                        'train': [train_data_attack, train_label_attack], 
                        'test': [test_data_attack, test_label_attack]
                        },
            "reconnaissance": 
                        {
                        'train': [train_data_reconnaissance, train_label_reconnaissance], 
                        'test': [test_data_reconnaissance, test_label_reconnaissance]
                        }
            }

# ---------------------------wandb-------------------------------
if args.use_wandb:
    import wandb                    
    os.environ["WANDB_API_KEY"] = "8f5046c50251d53a31a4248e1f1ca0dc4e5c8cf2"
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="ra",
        # Track hyperparameters and run metadata
        config={
            "data_fraction": args.data_fraction,
            "ps_epochs": args.ps_epochs,
            },
        name = f"df-{args.data_fraction}-{args.ps_epochs}"
        )

# ----------------create per-step detectors----------------------
ps_attack = Lstm("ps-detector-attack")
ps_attack.add_dataset(all_data['attack']) 

ps_recon = Lstm("ps-detector-recon")
ps_recon.add_dataset(all_data['reconnaissance']) 

ps_infec = Lstm("ps-detector-infec")
ps_infec.add_dataset(all_data['infection']) 

# ----------------train per-step detectors----------------------
metrics_dict_train = {}
metrics_dict_test = {}
for detector in [ps_attack, ps_recon, ps_infec]:
    #train data
    train_data = detector.dataset['train']
    train_examples = train_data[0]
    train_labels = train_data[1]

    #test data
    test_data = detector.dataset['test']
    test_examples = test_data[0]
    test_labels = test_data[1]
        
    features_len = train_examples.shape[1]
    print('features len is ', features_len)
    
    print_header("Training {} detector".format(detector.name))
    detector.learning(features_len, train_examples, train_labels, kind='', epochs=args.ps_epochs, patience=args.patience, fraction=args.data_fraction)

    print_header("Measureing {} detector performance on train data".format(detector.name))
    _, _, metrics_dict_ps_train = detector.detection(train_examples, train_labels, kind='')
    print("Metrics: \n", metrics_dict_ps_train)
    metrics_dict_train[detector.name] = metrics_dict_ps_train

    print_header("Measureing {} detector performance on test data".format(detector.name))
    _, _, metrics_dict_ps_test = detector.detection(test_examples, test_labels, kind='')
    print("Metrics: \n", metrics_dict_ps_test)
    metrics_dict_test[detector.name] = metrics_dict_ps_test

if args.use_wandb:
    keys = ('auprc', 'r_at_99p', 'r_at_95p', 'p_at_99r', 'p_at_95r')
    wandb_dict_train = dict((k, metrics_dict_train[ps_infec.name][k]) for k in keys)
    wandb_dict_test = dict((k, metrics_dict_test[ps_infec.name][k]) for k in keys)
    wandb.log({'train': wandb_dict_train, 'test':  wandb_dict_test})

metrics_dict_per_round = []
for r in range(args.relabel_rounds):

    # -----------------seq2seq stage----------------------
    #get events
    events = get_events(ps_attack, ps_recon, ps_infec, all_data['infection']['train'][0])

    #init seq2seq
    if not(args.freeze_seq2seq): 
        seq2seq = Seq2seqAttention('seq2seq')
    elif r == 0: 
        seq2seq = Seq2seqAttention('seq2seq')

    #train seq2seq
    seq2seq.learning(events, all_data['infection']['train'][1], seq2seq_config, epochs=args.s2s_epochs)

    #get seq2seq tagged events
    events_preds, tagged_seq2seq = seq2seq.analysis(events, all_data['infection']['train'][1], seq2seq_config)

    #get per-step infection detector tagged windows
    preds_ps_infec = ps_infec.predict(all_data['infection']['train'][0], kind='')
    preds_ps_infec = np.array(preds_ps_infec).squeeze()
    print('preds ps infec shape is', preds_ps_infec.shape)
    tagged_ps_infec = []
    for idx, pred in enumerate(preds_ps_infec):
        if pred>0.5:
            tagged_ps_infec.append(idx)


    # -----------------relabeling---------------------
    #strategy 1
    retrain_pos = []
    retrain_neg = []
    for idx in tagged_ps_infec:
        if idx in tagged_seq2seq:
            retrain_pos.append(idx)
        else:
            retrain_neg.append(idx)

    #overwritte original dataset with new positive and negative labels
    retrain_labels = copy.deepcopy(all_data['infection']['train'][1])
    retrain_data = copy.deepcopy(all_data['infection']['train'][0])
    for idx, l in enumerate(retrain_labels):
        if idx in retrain_pos:
            retrain_labels[idx] = 1
        if idx in retrain_neg:
            retrain_labels[idx] = 0


    # -----------------retrain per-step infection detector with new labels---------------------
    #def retrain_detector(detector, retrain_data, retrain_labels, test_data, test_labels):
    features_len = retrain_data.shape[1]
    print('features len is ', features_len)

    print_header("Retraining {} detector".format('infection'))
    ps_infec.learning(features_len, retrain_data, retrain_labels, kind='', epochs=args.ps_epochs)
                    
    print_header("Measureing {} detector performance on test data".format('infection'))
    _, _, metrics_dict_new = ps_infec.detection(detector.dataset['test'][0], detector.dataset['test'][1], kind='')

    print_header("Per-step infection detector metrics BEFORE relabeling")
    print( metrics_dict_test[ps_infec.name])

    print_header("Per-step infection detector metrics AFTER relabeling")
    metrics_dict_per_round.append(metrics_dict_new)
    for r, mdround in enumerate(metrics_dict_per_round):
        print(f'Round {r}:')
        print(mdround)

    if args.use_wandb:
        keys = ('auprc', 'r_at_99p', 'r_at_95p', 'p_at_99r', 'p_at_95r')
        wandb_dict_relabel = dict((k, metrics_dict_per_round[-1][k]) for k in keys)
        wandb.log({f'test_r{r}':  wandb_dict_relabel})

if args.use_wandb:
    wandb.finish()