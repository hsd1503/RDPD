# -*- coding: utf-8 -*-
"""

"""
import os
import dill
import numpy as np
from util import read_pamap, read_ptbdb, read_mimic_diag, my_eval
from config import *
from time import gmtime, strftime
from model import BaseCRNN

import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

## ------------------------ train step ------------------------
def train(model, optimizer, X_train, Y_train, config):
    res = {}
    
    model.train()
    n_train = len(Y_train)
    
    pred_all = []
    pred_temp_all = []
    att_all = []
    batch_start_idx = 0
    batch_end_idx = 0
    loss_all = []
    while batch_end_idx < n_train:
        batch_end_idx = batch_end_idx + config['batch_size']
        if batch_end_idx >= n_train:
            batch_end_idx = n_train
        batch_input = Variable(torch.FloatTensor(X_train[batch_start_idx: batch_end_idx, :])).cuda()
        batch_gt = Variable(torch.LongTensor(Y_train[batch_start_idx: batch_end_idx])).cuda()
        batch_temperature = Variable(torch.FloatTensor([config['temperature']])).cuda()
        
        pred, pred_temp, att = model(batch_input, batch_temperature)
        pred_all.append(pred.cpu().data.numpy())
        pred_temp_all.append(pred_temp.cpu().data.numpy())
        att_all.append(att.cpu().data.numpy())
        
        loss = torch.nn.NLLLoss()(torch.log(pred_temp), batch_gt)
            
        loss_all.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_start_idx = batch_start_idx + config['batch_size']

    loss_res = np.mean(loss_all)
    pred_all = np.concatenate(pred_all, axis=0)
    pred_temp_all = np.concatenate(pred_temp_all, axis=0)
    att_all = np.concatenate(att_all, axis=0)
    
    if config['model_type'] == 'teacher':
        res['eval'] = my_eval(Y_train, pred_temp_all)
    else:
        res['eval'] = my_eval(Y_train, pred_all)

    res['Y_train_soft'] = pred_temp_all
    res['att'] = att_all
    
    print('{0:.4f}, {1:.4f}'.format(loss_res, res['eval'][0]), end=' | ')

    return res

def train_student(model, optimizer, X_train, Y_train, Y_train_soft, att_teacher, config):
    res = {}
    
    model.train()
    n_train = len(Y_train)
    
    pred_all = []
    pred_temp_all = []
    pred_final_all = []
    att_all = []
    batch_start_idx = 0
    batch_end_idx = 0
    loss_all = []
    while batch_end_idx < n_train:
        batch_end_idx = batch_end_idx + config['batch_size']
        if batch_end_idx >= n_train:
            batch_end_idx = n_train
        batch_input = Variable(torch.FloatTensor(X_train[batch_start_idx: batch_end_idx, :])).cuda()
        batch_gt = Variable(torch.LongTensor(Y_train[batch_start_idx: batch_end_idx])).cuda()
        batch_Y_train_soft = Variable(torch.FloatTensor(Y_train_soft[batch_start_idx: batch_end_idx])).cuda()
        batch_att_teacher = Variable(torch.FloatTensor(att_teacher[batch_start_idx: batch_end_idx])).cuda()
        batch_temperature = Variable(torch.FloatTensor([config['temperature']])).cuda()
        
        pred, pred_temp, att = model(batch_input, batch_temperature)
        pred_final = torch.softmax(model.w1 * pred + model.w2 * pred_temp + model.b, dim=-1)
        
        pred_all.append(pred.cpu().data.numpy())
        pred_temp_all.append(pred_temp.cpu().data.numpy())
        pred_final_all.append(pred_final.cpu().data.numpy())
        att_all.append(att.cpu().data.numpy())

        # gt loss
        loss_1 = torch.nn.NLLLoss()(torch.log(pred), batch_gt)
        # soft loss, not use NLLLoss because targets are soft
        loss_2 = -1 * (config['temperature'])**2 * torch.sum(torch.mul(torch.log(pred_temp), batch_Y_train_soft)) / config['batch_size']
        # att loss [the input given is expected to contain log-probabilities]
        loss_3 = torch.nn.KLDivLoss(reduction='sum')(torch.log(att), batch_att_teacher) / config['batch_size']
        # combine pred loss
        loss_4 = torch.nn.NLLLoss()(torch.log(pred_final), batch_gt)
                
        loss = loss_1 + loss_2 + loss_3 + loss_4
            
        loss_all.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_start_idx = batch_start_idx + config['batch_size']

    loss_res = np.mean(loss_all)
    pred_all = np.concatenate(pred_all, axis=0)
    pred_temp_all = np.concatenate(pred_temp_all, axis=0)
    att_all = np.concatenate(att_all, axis=0)
    
    res['att'] = att_all
    res['eval'] = my_eval(Y_train, pred_all)
    res['pred'] = pred_all
    res['pred_final'] = pred_final_all
    res['stacking'] = [model.w1.cpu().data.numpy(), model.w2.cpu().data.numpy(), model.b.cpu().data.numpy()]
    
    print('{0:.4f}, {1:.4f}'.format(loss_res, res['eval'][0]), end=' | ')

    return res

def test(model, X_test, Y_test, config):
    
    res = {}
    
    model.eval()
    n_test = len(Y_test)
    
    pred_all = []
    batch_start_idx = 0
    batch_end_idx = 0
    while batch_end_idx < n_test:
        batch_end_idx = batch_end_idx + config['batch_size']
        if batch_end_idx >= n_test:
            batch_end_idx = n_test
        batch_input = Variable(torch.FloatTensor(X_test[batch_start_idx: batch_end_idx, :])).cuda()
        batch_gt = Variable(torch.LongTensor(Y_test[batch_start_idx: batch_end_idx])).cuda()
        
        if config['model_type'] == 'teacher':
            batch_temperature = Variable(torch.FloatTensor([config['temperature']])).cuda()
        else:
            batch_temperature = Variable(torch.FloatTensor([1])).cuda()

        pred, pred_temp, att = model(batch_input, batch_temperature)
        pred_all.append(pred.cpu().data.numpy())

        batch_start_idx = batch_start_idx + config['batch_size']

    pred_all = np.concatenate(pred_all, axis=0)
    
    res['eval'] = my_eval(Y_test, pred_all)
    res['pred'] = pred_all
    print('{0:.4f}'.format(res['eval'][0]), end=' | ')

    return res

## ------------------------ run ------------------------
def run_teacher(X_train, X_val, X_test, Y_train, Y_val, Y_test, config):
    
    res = {'Y_train_soft':[], 'att':[], 'eval_train':[], 'eval_val':[], 'eval_test':[], 'pred_val':[], 'pred_test':[]}

    my_model = BaseCRNN(config)
    my_model.cuda()
    optimizer = optim.Adam(my_model.parameters())
    
    print('loss, train AUROC | val AUROC | test AUROC')
    for epoch in range(n_epoch):
        res_train = train(my_model, optimizer, X_train, Y_train, config)
        res_val = test(my_model, X_val, Y_val, config)
        res_test = test(my_model, X_test, Y_test, config)
        print()
    
        res['Y_train_soft'].append(res_train['Y_train_soft'])
        res['att'].append(res_train['att'])
        res['eval_train'].append(res_train['eval'])
        res['eval_val'].append(res_val['eval'])
        res['eval_test'].append(res_test['eval'])
        res['pred_val'].append(res_val['pred'])
        res['pred_test'].append(res_test['pred'])
        
    res['eval_train'] = np.array(res['eval_train'])
    res['eval_val'] = np.array(res['eval_val'])
    res['eval_test'] = np.array(res['eval_test'])
    res['pred_val'] = np.array(res['pred_val'])
    res['pred_test'] = np.array(res['pred_test'])
        
    return res

def run_student(X_train, X_val, X_test, Y_train, Y_train_soft, att_teacher, Y_val, Y_test, config):
    
    res = {'att':[], 'stacking':[], 'eval_train':[], 'eval_val':[], 'eval_test':[], 'pred_val':[], 'pred_test':[], 'pred_final_train':[]}

    my_model = BaseCRNN(config)
    my_model.cuda()
#     for parameter in my_model.parameters():
#         print(parameter)
    optimizer = optim.Adam(my_model.parameters())
    
    print('loss, train AUROC | val AUROC | test AUROC')
    for epoch in range(n_epoch):
        res_train = train_student(my_model, optimizer, X_train, Y_train, Y_train_soft, att_teacher, config)
        res_val = test(my_model, X_val, Y_val, config)
        res_test = test(my_model, X_test, Y_test, config)
        print()

        res['att'].append(res_train['att'])
        res['stacking'].append(res_train['stacking'])
        res['eval_train'].append(res_train['eval'])
        res['eval_val'].append(res_val['eval'])
        res['eval_test'].append(res_test['eval'])
        res['pred_val'].append(res_val['pred'])
        res['pred_val'].append(res_val['pred'])
        res['pred_final_train'].append(res_train['pred_final'])
        
    res['eval_train'] = np.array(res['eval_train'])
    res['eval_val'] = np.array(res['eval_val'])
    res['eval_test'] = np.array(res['eval_test'])
    res['pred_val'] = np.array(res['pred_val'])
    res['pred_test'] = np.array(res['pred_test'])
    res['pred_final_train'] = np.array(res['pred_final_train'])
    
    return res


## ------------------------ main ------------------------
if __name__ == '__main__':
    
    try:
        os.stat('res')
    except:
        os.mkdir('res')
    
    dataset = 'mimic_diag'  # or 'pamap' or 'ptbdb'

    is_budget_save = True

    is_restore = False
    if is_restore:
        restore_run_id = ''
        with open('res/{0}.pkl'.format(restore_run_id), 'rb') as fin:
            restore_res = dill.load(fin)

    suffix = 'mimic'
    run_id = '{0}_{1}'.format(strftime("%Y%m%d_%H%M%S", gmtime()), suffix)

    ### ---------------------------- hyper-parameters ----------------------------

    n_epoch = 10
    n_run = 1
    temperature_list = [5]
    data_typ_list = list(range(n_run))

    ### poor data modalitites
    if dataset == 'pamap':
        view_list = [list(range(1,18)), list(range(18,35)), list(range(35,52))]
    elif dataset == 'ptbdb':
        view_list = [[0]]
    elif dataset == 'mimic_diag':
        view_list = [[0], [1], [3,4]]
    
    ### ---------------------------- run ----------------------------
    
    res = []
    
    for i_run in range(n_run):
        tmp_res = {}
        print("=="*40)
        print(i_run)

        for i_data_typ in data_typ_list:

            if dataset == 'pamap':
                X_train, X_val, X_test, Y_train, Y_val, Y_test = read_pamap(i_data_typ)
            elif dataset == 'ptbdb':
                X_train, X_val, X_test, Y_train, Y_val, Y_test = read_ptbdb(i_data_typ)
            elif dataset == 'mimic_diag':
                X_train, X_val, X_test, Y_train, Y_val, Y_test = read_mimic_diag(i_data_typ)

            for temperature in temperature_list:

                if dataset == 'pamap':
                    config_teacher = config_pamap_teacher
                    config_teacher['temperature'] = temperature
                    config_student = config_pamap_student_light
                    config_student['temperature'] = temperature
                elif dataset == 'ptbdb':
                    config_teacher = config_ptbdb_teacher
                    config_teacher['temperature'] = temperature
                    config_student = config_ptbdb_student_light
                    config_student['temperature'] = temperature
                elif dataset == 'mimic':
                    config_teacher = config_mimic_teacher
                    config_teacher['temperature'] = temperature
                    config_student = config_mimic_student_light
                    config_student['temperature'] = temperature
                elif dataset == 'mimic_diag':
                    config_teacher = config_mimic_diag_teacher
                    config_teacher['temperature'] = temperature
                    config_student = config_mimic_diag_student_light
                    config_student['temperature'] = temperature

                # run teacher
                res_id = 'teacher_{0}_{1}'.format(i_data_typ, temperature)
                if is_restore and res_id in restore_res:
                    res_teacher = restore_res[i_run][res_id]
                else:
                    res_teacher = run_teacher(X_train, X_val, X_test, Y_train, Y_val, Y_test, config_teacher)
                # use AUROC to select best soft label
                best_idx = np.argmax(res_teacher['eval_val'][:, 0])
                Y_train_soft = res_teacher['Y_train_soft'][best_idx]
                att_teacher = res_teacher['att'][best_idx]
                print('res_id', res_id, 'best_idx', best_idx, res_teacher['eval_test'][best_idx])
                tmp_res[res_id] = res_teacher

                ### modalitites
                for view in view_list:
                    
                    config_student['n_channel'] = len(view)

                    X_train_sub, X_val_sub, X_test_sub = X_train[:, :, view], X_val[:, :, view], X_test[:, :, view]
                    
                    # run student
                    res_id = 'student_{0}_{1}_{2}'.format(i_data_typ, temperature, min(view))
                    res_student = run_student(X_train_sub, X_val_sub, X_test_sub, Y_train, Y_train_soft, att_teacher, Y_val, Y_test, config_student)
                    best_idx = np.argmax(res_student['eval_val'][:, 0])
                    print('res_id', res_id, 'best_idx', best_idx, res_student['eval_test'][best_idx])
                    tmp_res[res_id] = res_student
                    
        if is_budget_save:
            for k in tmp_res:
                del tmp_res[k]['att']
        res.append(tmp_res)
        with open('res/{0}.pkl'.format(run_id), 'wb') as fout:
            dill.dump(res, fout)
