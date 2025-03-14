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
#import string
#import re
import random
import math 
import time 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 
import sklearn.metrics as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
#Rename afterwards
from EHRDataloader import iter_batch2
try:
    from termcolor import colored
except:
    def colored(string, color): return(string)

use_cuda = torch.cuda.is_available()

###### minor functions, plots and prints
#loss plot
def showPlot(points):
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.show()
    
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)    

#print to file function
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()

##### ML & Eval functions
def extra_metrics(yreal,yhat):
    ap_score= m.average_precision_score(yreal,yhat)
    tn, fp, fn, tp = m.confusion_matrix(yreal, (np.array(yhat)>0.5)).ravel()
    class_rep_dic=m.classification_report(yreal, (np.array(yhat)>0.5), output_dict=True,digits=4)
    return ap_score,tn, fp, fn, tp,class_rep_dic

def ml_evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    pred_prob=model.predict_proba(test_features)
    auc_p=roc_auc_score(test_labels,pred_prob[:,1])
    ap_score= m.average_precision_score(test_labels,pred_prob[:,1])
    tn, fp, fn, tp = m.confusion_matrix(test_labels, predictions).ravel()
    class_rep_dic=m.classification_report(test_labels,predictions,digits=4, output_dict=True)
    print('Model Performance')
    print('AUC = {:0.2f}%.'.format(auc_p*100))
    print('Confusion Matrix tn, fp, fn, tp:',tn, fp, fn, tp )
    print('Classification Report :',class_rep_dic)
    return test_labels,pred_prob[:,1],auc_p,ap_score,tn, fp, fn, tp,class_rep_dic#test_labels,pred_prob[:,1]


def get_val_loss(sample, label_tensor, seq_l, mtd, model):
    model.eval()
    with torch.no_grad():
        output = model(sample, seq_l, mtd)
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
def trainsample(sample, label_tensor, seq_l, mtd, model, optimizer): 
    model.train() ## LR added Jul 10, that is the right implementation
    model.zero_grad()
    
    output = model(sample, seq_l, mtd)
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
def trainbatches(train_mbs_list, val_mbs_list, model, optimizer, shuffle = True):#,we dont need this print print_every = 10, plot_every = 5): 
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
    for i, batch in enumerate(train_mbs_list):
        fids, sample, label_tensor, seq_l, mtd = batch
        case_pcts.append(sum(label_tensor)/len(label_tensor))
        n_case.append(sum(label_tensor))
        #print(f"cases {label_tensor.sum()} controls {(label_tensor==0).sum()}")
        output, loss = trainsample(sample, label_tensor, seq_l, mtd, model, optimizer) ### LR amended Sep 30 2020 to make sure we can change the loss function for survival
        train_loss += loss
        n_iter +=1
        if n_iter % plot_every == 0:
            train_losses.append(train_loss/plot_every)
            train_loss = 0  
    print(f"train average cases: {(np.mean(case_pcts)*100).round()}% (min N={np.min(n_case)})")
    ######### validation loop
    n_iter = 0   
    val_loss = 0
    val_losses = []
    case_pcts = []
    n_case = []
    for i, batch in enumerate(val_mbs_list):
        fids, sample, label_tensor, seq_l, mtd = batch
        #print(f"cases {label_tensor.sum()} controls {(label_tensor==0).sum()}")
        case_pcts.append(sum(label_tensor)/len(label_tensor))
        n_case.append(sum(label_tensor))
        loss = get_val_loss(sample, label_tensor, seq_l, mtd, model)
        val_loss += loss
        n_iter +=1
        if n_iter % plot_every == 0:
            val_losses.append(val_loss/plot_every)
            val_loss = 0
    print(f"valid average cases: {(np.mean(case_pcts)*100).round()}% (min N={np.min(n_case)})")

    return train_losses, val_losses 


##### utils for using already trained models

def get_preds(model, mbs_list):
    all_labels=[]
    all_fids=[]
    all_scores=[]
    for batch in mbs_list:
        fids, sample, label_tensor, seq_l, mtd = batch
        with torch.no_grad(): output = torch.sigmoid(model(sample, seq_l, mtd))
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
            fids, sample, label_tensor, seq_l, mtd = batch
            output = torch.sigmoid(model(sample, seq_l, mtd))
            crnt_out = pd.DataFrame({"FINNGENID": fids, "out": output.detach().numpy()})
            y_hat = pd.concat([y_hat, crnt_out])
            crnt_true = pd.DataFrame({"FINNGENID": fids, "true": label_tensor.cpu().data.view(-1).numpy()})
            y_real = pd.concat([y_real, crnt_true])
    y_total = pd.merge(y_real, y_hat, on="FINNGENID", how="left")
    y_total = y_total.groupby("FINNGENID").head(1)# in training date will have multiple occurances of same individual, keeping a random first one
    auc = roc_auc_score(y_total.true, y_total.out)
    
    return auc, y_total.true, y_total.out 
    
#define the final epochs running, use the different names

def epochs_run(epochs, train, valid, test, model, optimizer, trial=None, shuffle = True, which_model = 'RNN', patience = 20, model_out=""):  
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
        train_auc, _, _ = calculate_auc(model = model, mbs_list = train, which_model = which_model, shuffle = False)
        valid_auc, _, _ = calculate_auc(model = model, mbs_list = valid, which_model = which_model, shuffle = False)
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
                best_test_auc, _, _ = calculate_auc(model = model, mbs_list = test, which_model = which_model, shuffle = False)
                print(colored(f'\nValid AUC: {round(best_val_auc, 2)}, Test_AUC: {round(best_test_auc, 2)}, Test_eval_time: {timeSince(testeval_start)}\n', "yellow"))

        if ep - best_val_ep > patience: break
    #model.load_state_dict(best_model_state)
    return best_train_loss, best_val_loss, best_val_auc, best_test_auc, best_val_ep, model


def plot_roc_curve(label,score):
    fpr, tpr, ths = m.roc_curve(label, score) ### If I round it gives me an AUC of 64%
    roc_auc = m.auc(fpr, tpr)
    ### add aditional measures
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_roc_curve_combined_m(title,list_of_tuples):

    plt.figure()
    for tup in list_of_tuples:
        model_name, true_label, pred_score=tup
        fpr, tpr, ths = m.roc_curve(true_label, pred_score) ### If I round it gives me an AUC of 64%
        roc_auc = m.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label='%s (AUC = %0.1f%%)'%(model_name,roc_auc*100))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title(title,fontdict={'fontsize':15,'fontweight' :800} )
    plt.legend(loc="lower right",fontsize=14)
    plt.show()

#### Calibration Plot
from sklearn.calibration import calibration_curve
def plot_calibration_curve(name, fig_index,y_test, probs):
    """Plot calibration curve for est w/o and with calibration. """

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    frac_of_pos, mean_pred_value = calibration_curve(y_test, probs, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f'{name}')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot ({name})')
    
    ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
 
def eval_metrics(yreal,yhat,th=0.5):
    auc_score= m.roc_auc_score(yreal,yhat)
    ap_score= m.average_precision_score(yreal,yhat)
    tn, fp, fn, tp = m.confusion_matrix(yreal, (np.array(yhat)>=th)).ravel()
    class_rep_dic= m.classification_report(yreal, (np.array(yhat)>=th), output_dict=True,digits=4)
    return auc_score,ap_score,tn, fp, fn, tp,class_rep_dic



def get_best_thre(true_label, pred_score,verbose=True):
    fpr, tpr, ths = m.roc_curve(true_label, pred_score) ### If I round it gives me an AUC of 64%
    auc= m.auc(fpr, tpr)
    dist=np.sqrt((1-tpr)**2+(fpr)**2)
    optimalindex=np.argmin(dist)
    if verbose:
      print('fpr:',fpr, ', tpr:' ,tpr, ', ths :',ths,', auc:',auc,', optimalindex', optimalindex)    
      print ("optimalindex,dist[optimalindex],ths[optimalindex],tpr[optimalindex],fpr[optimalindex]")
      print (optimalindex,dist[optimalindex],ths[optimalindex],tpr[optimalindex],fpr[optimalindex])
    return ths[optimalindex]

def metrics_on_sens(sens,label,preds):
    fpr, tpr, ths = m.roc_curve(label,preds)
    auc_score = m.roc_auc_score(label,preds)
    x=np.array(np.where(tpr>=sens)).min()
    print ('Model AUC : ',auc_score)
    print ('Threshold for ',sens,' sensitivity : ', ths[x])
    print ('Sensitivity : ',tpr[x])
    print ('Specificity : ',1-fpr[x])

def metrics_on_ths(th,label,preds):
    fpr, tpr, ths = m.roc_curve(label,preds)
    auc_score = m.roc_auc_score(label,preds)
    x=np.array(np.where(ths>=th)).max()
    print ('Model AUC : ',auc_score)
    print ('Threshold ', ths[x])
    print ('Sensitivity : ',tpr[x])
    print ('Specificity : ',1-fpr[x])
