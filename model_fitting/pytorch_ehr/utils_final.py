# -*- coding: utf-8 -*-
"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo
@authors: Lrasmy , Jzhu @ DeguiZhi Lab - UTHealth SBMI
Last revised Feb 20 2020

Changed by KE Detrois

Last updated March 14 2025
"""
from __future__ import print_function, division
import sys
sys.path.append(("/home/ivm/valid/scripts/pytorch_ehr/"))
from io import open
import random
import math 
import time 
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score  
import numpy as np
import pandas as pd
import optuna
try:
    from termcolor import colored
except:
    def colored(string, color): return(string)

use_cuda = torch.cuda.is_available()

def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)    

def get_val_loss(sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor, model):
    model.eval()
    with torch.no_grad():
        output = model(sample, seq_l, mtd, ages_tensor, sexs_tensor)
        if label_tensor.shape[-1]>1 :
            criterion = nn.CrossEntropyLoss(reduction='sum')            
            mc = output.shape[-1]
            e,d = label_tensor.squeeze().T
            d_m = d
            d_m[d_m > mc-3] = mc-2
            d_m[e==0] = mc-1      
            if use_cuda: lnt_typ=torch.cuda.LongTensor
            else: lnt_typ=torch.LongTensor
            loss = criterion(output, d_m.view(-1).type(lnt_typ))
        else:
            class_weights = torch.tensor([(label_tensor==0.).sum()/label_tensor.sum()], dtype=torch.float64)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(output.squeeze(), label_tensor.squeeze())
    return loss.item()

###### major model training utilities
def trainsample(sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor, model, optimizer): 
    model.train() ## LR added Jul 10, that is the right implementation
    model.zero_grad()
    
    output = model(sample, seq_l, mtd, ages_tensor, sexs_tensor)
    if label_tensor.shape[-1]>1 :
        criterion = nn.CrossEntropyLoss(reduction='sum')
        mc = output.shape[-1]
        e,d = label_tensor.squeeze().T
        d_m = d
        d_m[d_m > mc-3] = mc-2
        d_m[e==0] = mc-1      
        if use_cuda: lnt_typ=torch.cuda.LongTensor
        else: lnt_typ=torch.LongTensor
        loss = criterion(output, d_m.view(-1).type(lnt_typ))
    else:
        class_weights = torch.tensor([(label_tensor==0.).sum()/label_tensor.sum()], dtype=torch.float64)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output.squeeze(), label_tensor.squeeze())  ##LR 9/2/21 added squeez for compatability with Pytorch1.7
        
    loss.backward()   
    optimizer.step()
    return output, loss.item()


#train with loaders
def trainbatches(train_mbs_list, 
                 val_mbs_list, 
                 model, 
                 optimizer, 
                 shuffle = True):#,we dont need this print print_every = 10, plot_every = 5): 
    train_loss = 0
    train_losses = []
    plot_every = 1
    n_iter = 0 
    if shuffle: 
         # you can also shuffle batches using iter_batch2 method in EHRDataloader
        #  loader = iter_batch2(mbs_list, len(mbs_list))
        random.shuffle(train_mbs_list)
        #random.shuffle(val_mbs_list)
    ######### training loop
    case_pcts = []
    n_case = []
    case_fids = set()
    for i, batch in enumerate(train_mbs_list):
        fids, sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor = batch
        case_pcts.append(sum(label_tensor)/len(label_tensor))
        n_case.append(sum(label_tensor))
        case_fids = case_fids.union(set([fid for fid, label in zip(fids, label_tensor.cpu().data.view(-1).numpy()) if label==1]))
        output, loss = trainsample(sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor, model, optimizer) 
        train_loss += loss
        n_iter +=1
        if n_iter % plot_every == 0:
            train_losses.append(train_loss/plot_every)
            train_loss = 0  
    print(f"train total cases: {len(case_fids)}")
    print(f"train average cases: {(np.mean(case_pcts)*100).round(2)}% (min N={np.min(n_case)})")
    ######### validation loop
    n_iter = 0   
    val_loss = 0
    val_losses = []
    case_pcts = []
    n_case = []
    case_fids = set()

    for i, batch in enumerate(val_mbs_list):
        fids, sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor = batch
        #print(f"cases {label_tensor.sum()} controls {(label_tensor==0).sum()}")
        case_pcts.append(sum(label_tensor)/len(label_tensor))
        n_case.append(sum(label_tensor))
        case_fids = case_fids.union(set([fid for fid, label in zip(fids, label_tensor.cpu().data.view(-1).numpy()) if label==1]))

        loss = get_val_loss(sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor, model)
        val_loss += loss
        n_iter +=1
        if n_iter % plot_every == 0:
            val_losses.append(val_loss/plot_every)
            val_loss = 0
    print(f"valid total cases: {len(case_fids)}")
    print(f"valid average cases: {(np.mean(case_pcts)*100).round(2)}% (min N={np.min(n_case)})")

    return train_losses, val_losses 


##### utils for using already trained models

def get_preds(model, mbs_list):
    all_labels=[]
    all_fids=[]
    all_scores=[]
    for batch in mbs_list:
        fids, sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor = batch
        with torch.no_grad(): output = torch.sigmoid(model(sample, seq_l, mtd, ages_tensor, sexs_tensor))
        all_labels.extend(label_tensor.cpu().data.view(-1).numpy())
        all_fids.extend(fids)
        all_scores.extend(output.detach().numpy())
    return all_fids, all_labels, all_scores
    
def calculate_auc(model, mbs_list, which_model = 'RNN', shuffle = True):
    model.eval() ## LR added Jul 10, that is the right implementation
    y_real = pd.DataFrame()
    y_hat = pd.DataFrame()
    if shuffle: 
        random.shuffle(mbs_list)
    with torch.no_grad():
        for i,batch in enumerate(mbs_list):
            fids, sample, label_tensor, seq_l, mtd, ages_tensor, sexs_tensor = batch
            output = torch.sigmoid(model(sample, seq_l, mtd, ages_tensor, sexs_tensor))
            crnt_out = pd.DataFrame({"FINNGENID": fids, "out": output.detach().numpy()})
            y_hat = pd.concat([y_hat, crnt_out])
            crnt_true = pd.DataFrame({"FINNGENID": fids, "true": label_tensor.cpu().data.view(-1).numpy()})
            y_real = pd.concat([y_real, crnt_true])
    y_total = pd.merge(y_real, y_hat, on="FINNGENID", how="left")
    y_total = y_total.groupby("FINNGENID").head(1)# in training date will have multiple occurances of same individual, keeping a random first one
    auc = roc_auc_score(y_total.true, y_total.out)
    
    return auc, y_total.true, y_total.out 
    
#define the final epochs running, use the different names

def epochs_run(epochs, 
               train, 
               valid, 
               test, 
               model, 
               optimizer, 
               trial=None, 
               model_name = 'RNN', 
               early_stop = 20, 
               model_out=""):  
    best_val_loss = 10.0
    best_train_loss = 10.0
    best_val_ep = 0
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_model_state = None
    
    for ep in range(epochs):
        start = time.time()
        train_losses, val_losses = trainbatches(train, valid, model=model, optimizer = optimizer)
        train_time = timeSince(start)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        #### pruning not promising trials
        if trial is not None:
            trial.report(avg_val_loss, ep)
            if trial.should_prune():
                print(avg_val_loss)
                raise optuna.TrialPruned()
        valid_start = time.time()
        train_auc, _, _ = calculate_auc(model = model, mbs_list = train, which_model = model_name, shuffle = False)
        valid_auc, _, _ = calculate_auc(model = model, mbs_list = valid, which_model = model_name, shuffle = False)
        valid_time = timeSince(valid_start)
        print(colored(f'Epoch {ep}: Train AUC: {round(train_auc, 2)}, Valid AUC: { round(valid_auc, 2)}, Training average loss: {round(avg_train_loss, 4)}, Valid average loss: {round(avg_val_loss, 4)}, Train time: {train_time}, Eval time: {valid_time}', 'green'))
        if avg_val_loss < best_val_loss: 
            best_train_loss = avg_train_loss
            best_val_loss = avg_val_loss
            best_val_auc = valid_auc
            best_val_ep = ep
            if model_out != "":
                torch.save(model, model_out + ".pth")
                torch.save(model.state_dict(), model_out + ".st")
            if test:
                testeval_start = time.time()
                best_test_auc, _, _ = calculate_auc(model = model, mbs_list = test, which_model = model_name, shuffle = False)
                print(colored(f'\nValid AUC: {round(best_val_auc, 2)}, Test_AUC: {round(best_test_auc, 2)}, Test_eval_time: {timeSince(testeval_start)}\n', "yellow"))

        if ep - best_val_ep > early_stop: break
    #model.load_state_dict(best_model_state)
    return best_val_loss, best_val_ep, model