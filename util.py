# -*- coding: utf-8 -*-
"""

"""

import dill
import numpy as np
from scipy import interpolate
import copy
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

## ---------------------------- evaluation ----------------------------

def label_ont_hot(label):
    label_list = list(np.unique(label))
    out_label = []
    for l in label:
        tmp_label = [0 for _ in range(len(label_list))]
        tmp_label[label_list.index(l)] = 1
        out_label.append(tmp_label)
    return np.array(out_label)

def my_eval(y_true_idx, y_pred_proba, verbose=False):
    """
    y_true, y_pred, y_pred_proba are all matrix
    """
    y_true = label_ont_hot(y_true_idx)
    y_pred = np.zeros_like(y_pred_proba)
    y_pred[np.arange(y_pred_proba.shape[0]), np.argmax(y_pred_proba, axis=1)] = 1
    ret = {}
    ret['auroc'] = roc_auc_score(y_true, y_pred_proba)
    ret['auprc'] = average_precision_score(y_true, y_pred_proba)
    ret['acc'] = accuracy_score(y_true, y_pred)
    ret['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    ret['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    if verbose:
        print(ret)
        print(cm)
    return list(ret.values())

## ---------------------------- read data PAMAP noise ----------------------------
def read_pamap_noise(amp):
    X_train, X_val, X_test, y_train, y_val, y_test = read_pamap(typ = 0)
    
    X_train += amp*(2*np.random.rand(X_train.shape[0], X_train.shape[1], X_train.shape[2]) - 1)
    X_val += amp*(2*np.random.rand(X_val.shape[0], X_val.shape[1], X_val.shape[2]) - 1)
    X_test += amp*(2*np.random.rand(X_test.shape[0], X_test.shape[1], X_test.shape[2]) - 1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

## ---------------------------- read data MIMIC ----------------------------
def read_mimic_diag(seed):
    with open('data/mimic_diag.pkl', 'rb') as fin:
        res = dill.load(fin)
    
    data = res['data']
    label = res['label']

    ### split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
    print('train', Counter(y_train))
    print('val', Counter(y_val))
    print('test', Counter(y_test))
    
    return X_train, X_val, X_test, y_train, y_val, y_test


## ---------------------------- read data PAMAP2 and PTBDB ----------------------------
def print_stat_counter(ts):
    cnter = Counter(ts)
    for k, v in cnter.items():
        print(k, v)
        
def read_pamap(typ = 0):
    with open('data/pamap.pkl', 'rb') as fin:
        res = dill.load(fin)
    data = res['data']
    label = res['label']
    sid = res['sid']
    print_stat_counter(sid)
    
    data[np.isnan(data)] = 0.0
    
    selected_samples = (label[:, 0] != 1)
    data = data[selected_samples]
    label = label[selected_samples]
    sid = sid[selected_samples]
    
    # shuffle
    np.random.seed(seed=typ)
    shuffle_idx = np.random.permutation(len(sid))
    data = data[shuffle_idx]
    label = label[shuffle_idx]
    sid = sid[shuffle_idx]    
    
    if typ == 0:
        s1 = 'subject101'
        s2 = 'subject105'
    elif typ == 1:
        s1 = 'subject102'
        s2 = 'subject108'
    elif typ == 2:
        s1 = 'subject101'
        s2 = 'subject102'
        
    train_id = np.logical_and(sid != s1, sid != s2)
    val_id = (sid == s1)
    test_id = (sid == s2)
        
    X_train, X_val, X_test = data[train_id], data[val_id], data[test_id]
    y_train, y_val, y_test = label[train_id, 1:], label[val_id, 1:], label[test_id, 1:]
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    print('train', np.sum(y_train, axis=0))
    print('val', np.sum(y_val, axis=0))
    print('test', np.sum(y_test, axis=0))
    
    y_train = np.array([np.where(r==1)[0][0] for r in y_train])
    y_val = np.array([np.where(r==1)[0][0] for r in y_val])
    y_test = np.array([np.where(r==1)[0][0] for r in y_test])
    
    return X_train, X_val, X_test, y_train, y_val, y_test
        
def read_ptbdb(seed = 0):
        
    selected_label = [0,1,2,3,4,5]
    
    with open('data/ptbdb.pkl', 'rb') as fin:
        res = dill.load(fin)
    data = res['data']
    label = res['label']
    sid = res['sid']

    valid_idx = (np.sum(label[:,selected_label], axis=1) == 1)
    data = data[valid_idx]
    label = label[valid_idx]
    sid = sid[valid_idx]
    
    # shuffle
    np.random.seed(seed=seed)
    shuffle_idx = np.random.permutation(len(sid))
    data = data[shuffle_idx]
    label = label[shuffle_idx]
    sid = sid[shuffle_idx]

    all_sid = np.unique(sid)
    train_sid, test_sid = train_test_split(all_sid, test_size=0.2, random_state=seed)
    val_sid, test_sid = train_test_split(test_sid, test_size=0.5, random_state=seed)
    print(train_sid.shape, val_sid.shape, test_sid.shape)

    train_id, val_id, test_id = np.full(len(sid), False), np.full(len(sid), False), np.full(len(sid), False)
    for i in range(len(sid)):
        if sid[i] in train_sid:
            train_id[i] = True
        elif sid[i] in val_sid:
            val_id[i] = True
        elif sid[i] in test_sid:
            test_id[i] = True

    X_train, X_val, X_test = data[train_id], data[val_id], data[test_id]
    y_train, y_val, y_test = label[train_id, :][:, selected_label], label[val_id, :][:, selected_label], label[test_id, :][:, selected_label]
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    print('train', np.sum(y_train, axis=0))
    print('val', np.sum(y_val, axis=0))
    print('test', np.sum(y_test, axis=0))
    
    y_train = np.array([np.where(r==1)[0][0] for r in y_train])
    y_val = np.array([np.where(r==1)[0][0] for r in y_val])
    y_test = np.array([np.where(r==1)[0][0] for r in y_test])
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':

    X_train, X_val, X_test, y_train, y_val, y_test = read_mimic_diag()

#     X_train, X_val, X_test, y_train, y_val, y_test = read_pamap()
    
#     X_train, X_val, X_test, y_train, y_val, y_test = read_ptbdb()
    
        
        