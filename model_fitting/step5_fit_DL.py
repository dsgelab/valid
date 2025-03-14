# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo
@authors: Lrasmy , Jzhu  @ DeguiZhi Lab - UTHealth SBMI
Last revised Feb 20 2020
"""
from __future__ import print_function, division
from io import open
import string
import re
import random

import os
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
sys.path.append(("/home/ivm/valid/scripts/pytorch_ehr/"))
from model_eval_utils import create_report
import argparse
import time
import math

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from utils import *
from plot_utils import create_report_plot

try:
    import cPickle as pickle
except:
    import pickle

from EHRDataloader import EHRdataFromPickles, EHRdataloader
import utils_final as ut #:)))) 
from EHREmb import EHREmbeddings
import optuna
from models import EHR_RNN, EHR_DRNN, EHR_QRNN, EHR_TLSTM, EHR_LR_emb, RETAIN
from utils_final import get_preds
from model_eval_utils import eval_subset
#silly ones
try:
    from termcolor import colored
except:
    def colored(string, color): return(string)
from sklearn.metrics import precision_recall_curve

# check GPU availability
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

def get_obj(ehr_model, args, hyperparams):
    #model optimizers to choose from. Upper/lower case dont matter
    if hyperparams["optimizer"] == 'adam':
        optimizer = optim.Adam(ehr_model.parameters(), 
                               lr=1/(10*hyperparams["lr"]), 
                               weight_decay=hyperparams["L2"],
                               eps = args.eps)
    elif hyperparams["optimizer"]  == 'adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), 
                                   lr=1/(10*hyperparams["lr"]), 
                                   weight_decay=hyperparams["L2"],
                                   eps = args.eps)
    elif hyperparams["optimizer"]  == 'adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), 
                                  lr=1/(10*hyperparams["lr"]), 
                                  weight_decay=hyperparams["L2"]) 
    elif hyperparams["optimizer"]  == 'adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), 
                                 lr=1/(10*hyperparams["lr"]), 
                                 weight_decay=hyperparams["L2"],
                                 eps = args.eps)
    elif hyperparams["optimizer"] == 'asgd':
        optimizer = optim.ASGD(ehr_model.parameters(), 
                               lr=1/(10*hyperparams["lr"]), 
                               weight_decay=hyperparams["L2"])
    elif hyperparams["optimizer"]  == 'rmsprop':
        optimizer = optim.RMSprop(ehr_model.parameters(), 
                                  lr=1/(10*hyperparams["lr"]), 
                                  weight_decay=hyperparams["L2"],
                                  eps = args.eps)
    elif hyperparams["optimizer"]  == 'rprop':
        optimizer = optim.Rprop(ehr_model.parameters(), lr=1/(10*hyperparams["lr"]))
    elif hyperparams["optimizer"]  == 'sgd':
        optimizer = optim.SGD(ehr_model.parameters(), 
                              lr=1/(10*hyperparams["lr"]), 
                              weight_decay=hyperparams["L2"])
    else:
        raise NotImplementedError
    return(optimizer)

def get_model(args, hyperparams):
    if args.which_model == 'RNN': 
        ehr_model = EHR_RNN(input_size= args.input_size, 
                                  embed_dim=2**hyperparams["embed_dim_exp"], 
                                  hidden_size= 2**hyperparams["hidden_size_exp"],
                                  n_layers= hyperparams["n_layers"],
                                  dropout_r=0.1*hyperparams["dropout_r"],
                                  cell_type=args.cell_type,
                                  bii=args.bii,
                                  time=args.time,
                                  preTrainEmb= args.preTrainEmb) 
    elif args.which_model == 'DRNN': 
        ehr_model = EHR_DRNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=0.1*hyperparams["dropout_r"],
                                  cell_type=args.cell_type, #default ='DRNN'
                                  bii= False,
                                  time = args.time, 
                                  preTrainEmb= args.preTrainEmb)     
    elif args.which_model == 'QRNN': 
        ehr_model = EHR_QRNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=0.1*hyperparams["dropout_r"],
                                  cell_type= 'QRNN', #doesn't support normal cell types
                                  bii= False, #QRNN doesn't support bi
                                  time = args.time,
                                  preTrainEmb= args.preTrainEmb)  
    elif args.which_model == 'TLSTM': 
        ehr_model = EHR_TLSTM(input_size= args.input_size, 
                                  embed_dim=2**hyperparams["embed_dim_exp"], 
                                  hidden_size=2**hyperparams["hidden_size_exp"],
                                  n_layers=1,
                                  dropout_r=0.1*hyperparams["dropout_r"],
                                  cell_type= 'TLSTM', #doesn't support normal cell types
                                  bii=False, 
                                  time = args.time, 
                                  preTrainEmb=args.preTrainEmb)  
    elif args.which_model == 'RETAIN': 
        ehr_model = RETAIN(embed_dim=2**hyperparams["embed_dim_exp"], 
                                  hidden_size=2**hyperparams["hidden_size_exp"],
                                  n_layers=1,
                                  input_size=args.input_size)
    else: 
        ehr_model = EHR_LR_emb(input_size = args.input_size,
                                     embed_dim = args.embed_dim,
                                     preTrainEmb= args.preTrainEmb)

    return(ehr_model)

def optuna_objective(trial, train_mbs, valid_mbs, test_mbs, args):
    # Suggest hyperparameters

    params = {
        'embed_dim_exp': trial.suggest_int('embed_dim_exp', 2, 10, 1),
        'hidden_size_exp': trial.suggest_int('hidden_size_exp', 2, 10, 1),
        'dropout_r': trial.suggest_int("dropout_r", 0, 5, 1),
        'L2': trial.suggest_float('L2', 1e-6, 1e-2, log=True),
        'lr': trial.suggest_int('lr', 1, 11, 2),
        #'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'adamax', 'asgd', 'rmsprop', 'rprop', 'sgd']),
        'optimizer': trial.suggest_categorical('optimizer', ['adagrad', 'adamax']),

    }
    if args.which_model == 'RNN': params['n_layers'] = trial.suggest_int('n_layers', 1, 5)
    # Train model
    print(params)
    ehr_model = get_model(args, params)
    optimizer = get_obj(ehr_model, args, params)
    if use_cuda:
        ehr_model = ehr_model.cuda() 
    #######Step3. Train, validation and test. default: batch shuffle = true 
    try:
        best_train_loss, best_val_loss, best_val_auc, best_test_auc, best_val_ep, best_model = ut.epochs_run(args.epochs, 
                      train = train_mbs, 
                      valid = valid_mbs, 
                      test = test_mbs, 
                      model = ehr_model, 
                      optimizer = optimizer,
                      trial=trial,
                      shuffle = True, 
                      which_model = args.which_model, 
                      patience = args.patience)
        trial.set_user_attr("best_iteration", int(best_val_ep))
        if best_val_loss is not None:
            return best_val_loss
        else:
            raise optuna.TrialPruned()
    #we can keyboard interupt now 
    except KeyboardInterrupt:
        print(colored('-' * 89, 'green'))
        print(colored('Manual pruning','green'))
        raise optuna.TrialPruned()



def run_trials(train_mbs, valid_mbs, test_mbs, args, study_name):   
    """Runs the first step of the XGBoost optimization, which is to find the best hyperparameters for the model on a high learning rate.
       The function uses Optuna to optimize the hyperparameters. The function returns the best hyperparameters found.
       The function also logs the best hyperparameters found and the best boosting round."""     
    if args.refit: 
        try:
            optuna.delete_study(study_name=study_name, storage="sqlite:///" + args.lab_name + "_optuna.db")
        except KeyError:
            print("Record not found, have to create new anyways.")
    study = optuna.create_study(direction='minimize', 
                                sampler= optuna.samplers.TPESampler(seed=429), 
                                study_name=study_name, 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.patience),
                                storage="sqlite:///" + args.lab_name + "_optuna.db", 
                                load_if_exists=True)
    tic = time.time()
    timer = Timer()
    np.random.seed(9234)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    if args.n_trials == 1:
        while time.time() - tic < args.time_step1:
            study.optimize(lambda trial: optuna_objective(trial, train_mbs, valid_mbs, test_mbs, args) , n_trials=1)
    else:
        study.optimize(lambda trial: optuna_objective(trial, train_mbs, valid_mbs, test_mbs, args) , n_trials=args.n_trials)

    
    logging.info('Stage 1 ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info(f'best score = {study.best_trial.value}')
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {args.lr}')
    logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
    logging.info('best tree params --------------------------')
    for k, v in study.best_trial.params.items(): logging.info(str(k)+':'+str(v))
        
    return(study.best_trial.params)


def get_parser_arguments():
    #this is where you define all the things you wanna run in your main file
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')
    
    #EHRdataloader 
    parser.add_argument('-root_dir', type = str, default = '../data/' , help='the path to the folders with pickled file(s)')
    parser.add_argument("-lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("-res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step5_predict/1_year_buffer/")
    parser.add_argument("-pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("-time_step1", type=int, default=300)
    parser.add_argument("-n_trials", type=int, help="Number of random samples for bootstrapping of metrics", default=1)
    parser.add_argument("-refit", type=int, help="whether to redo optuna optimization or continue saved.", default=1)
    parser.add_argument("-file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument("-date_model_fit", type=str, default="")
    parser.add_argument("-n_boots", type=int, help="Number of random samples for bootstrapping of metrics", default=500)
    parser.add_argument("-save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)

    ### Kept original -files variable not forcing original unique naming for files
    parser.add_argument('-files', nargs='+', default = ['hf.train'], help='''the name(s) of pickled file(s), separtaed by space. so the argument will be saved as a list 
                        If list of 1: data will be first split into train, validation and test, then 3 dataloaders will be created.
                        If list of 3: 3 dataloaders will be created from 3 files directly. Please give files in this order: training, validation and test.''')
    parser.add_argument('-test_ratio', type = float, default = 0.2, help='test data size [default: 0.2]')
    parser.add_argument('-valid_ratio', type = float, default = 0.1, help='validation data size [default: 0.1]')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for training, validation or test [default: 128]')
    #EHRmodel
    parser.add_argument('-simple_fit', type = bool, default = False) 
    parser.add_argument('-skip_model_fit', type = bool, default = 0) 

    parser.add_argument('-which_model', type = str, default = 'DRNN',choices= ['RNN','DRNN','QRNN','TLSTM','LR','RETAIN'], help='choose from {"RNN","DRNN","QRNN","TLSTM","LR","RETAIN"}') 
    parser.add_argument('-cell_type', type = str, default = 'GRU', choices=['RNN', 'GRU', 'LSTM'], help='For RNN based models, choose from {"RNN", "GRU", "LSTM", "QRNN" (for QRNN model only)}, "TLSTM (for TLSTM model only')
    parser.add_argument('-input_size', nargs='+', type=int , default = [15817], help='''input dimension(s) separated in space the output will be a list, decide which embedding types to use. 
                        If len of 1, then  1 embedding; len of 3, embedding medical, diagnosis and others separately (3 embeddings) [default:[15817]]''')
    parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-hidden_size', type=int, default=128, help='size of hidden layers [default: 128]')
    parser.add_argument('-dropout_r', type=float, default=0.1, help='the probability for dropout[default: 0.1]')
    parser.add_argument('-n_layers', type=int, default=1, help='number of Layers, for Dilated RNNs, dilations will increase exponentialy with mumber of layers [default: 1]')
    parser.add_argument('-bii', type=bool, default=False, help='indicator of whether Bi-directin is activated. [default: False]')
    parser.add_argument('-time', type=bool, default=False, help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('-preTrainEmb', type= str, default='', help='path to pretrained embeddings file. [default:'']')
    parser.add_argument('-model_prefix', type = str, default = 'hf.train' , help='the prefix name for the saved model e.g: hf.train [default: [(training)file name]')
    # training 
    parser.add_argument('-lr', type=float, default=10**-2, help='learning rate [default: 0.01]')
    parser.add_argument('-L2', type=float, default=10**-4, help='L2 regularization [default: 0.0001]')
    parser.add_argument('-eps', type=float, default=10**-8, help='term to improve numerical stability [default: 0.00000001]')
    parser.add_argument('-epochs', type=int, default= 100, help='number of epochs for training [default: 100]')
    parser.add_argument('-patience', type=int, default= 5, help='number of stagnant epochs to wait before terminating training [default: 5]')
    parser.add_argument('-optimizer', type=str, default='adam', choices=  ['adam','adadelta','adagrad', 'adamax', 'asgd','rmsprop', 'rprop', 'sgd'], 
                        help='Select which optimizer to train [default: adam]. Upper/lower case does not matter') 
    #parser.add_argument('-cuda', type= bool, default=True, help='whether GPU is available [default:True]')
    args = parser.parse_args()
    return(args)


def get_crnt_preds(data_mbs, mdl, crnt_set=0):
    t2_fids, t2_labels, t2_scores = get_preds(mdl , data_mbs)
    crnt_preds = pd.DataFrame({"FINNGENID": t2_fids, "TRUE_ABNORM": t2_labels, "ABNORM_PROBS": t2_scores})
    precision_, recall_, proba = precision_recall_curve(crnt_preds.TRUE_ABNORM, crnt_preds.ABNORM_PROBS)
    optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
    logging.info(f"Optimal cut-off set {crnt_set} for prediction based on PR {optimal_proba_cutoff}")
    crnt_preds = crnt_preds.assign(ABNORM_PREDS = (crnt_preds.ABNORM_PROBS>optimal_proba_cutoff).astype("int64"))

    return(crnt_preds)
    
def get_out_data(model_final, train_mbs, valid_mbs, test_mbs, file_path_labels):
    ###### Predictions
    train_preds = get_crnt_preds(train_mbs, model_final, 0)
    test_preds = get_crnt_preds(test_mbs, model_final, 2)
    val_preds = get_crnt_preds(valid_mbs, model_final, 1)
    all_preds = pd.concat([train_preds, test_preds, val_preds])

    ####### Original data
    data = pd.read_csv(file_path_labels, sep=",")
    out_data = data[["FINNGENID", "EVENT_AGE", "SET"]]
    out_data = out_data.assign(TRUE_VALUE = data["y_MEAN"])
    out_data = pd.merge(out_data, all_preds, on="FINNGENID", how="inner")
    return(out_data)

def main():
    timer = Timer()
    args = get_parser_arguments()
    
    study_name = args.which_model + "_" + args.cell_type + "_" + args.pred_descriptor
    if args.bii: study_name += "_bii"
    if args.time: study_name += "_time"

    out_dir = args.res_dir + study_name + "/"
    out_name = args.lab_name 
    log_file_name = args.lab_name + "_" + args.pred_descriptor + "_preds_" + get_datetime()
    if args.date_model_fit == "": args.date_model_fit = get_date()

    make_dir(out_dir + "plots/"); make_dir(out_dir + "models/"); make_dir(out_dir + "down/"); make_dir(out_dir + "down/" + args.date_model_fit)    

    ####Step1. Data preparation
       
    print(colored("\nLoading and preparing data...", 'green'))
    if len(args.files) == 1:
        print('1 file found. Data will be split into train, validation and test.')
        data = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.files[0], 
                              sort= False,
                              test_ratio = args.test_ratio, 
                              valid_ratio = args.valid_ratio,
                              model=args.which_model) #No sort before splitting
    
        # Dataloader splits
        train, test, valid = data.__splitdata__() #this time, sort is true
        # can comment out this part if you dont want to know what's going on here
        print(colored("\nSee an example data structure from training data:", 'green'))
        print(data.__getitem__(35, seeDescription = True))
    elif len(args.files) == 2:
        print('2 files found. 2 dataloaders will be created for train and validation')
        train = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.files[0], 
                              sort= True,
                              model=args.which_model)
        valid = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.files[1], 
                              sort= True,
                              model=args.which_model)
        test = None
        
    else:
        print('3 files found. 3 dataloaders will be created for each')
        train = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.files[0], 
                              sort= True,
                              model=args.which_model)
        valid = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.files[1], 
                              sort= True,
                              model=args.which_model)
        test = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.files[2], 
                              sort= True,
                              model=args.which_model)
        print(colored("\nSee an example data structure from training data:", 'green'))
        print(train.__getitem__(40, seeDescription = True))
    

    print(colored("\nSample data lengths for train, validation and test:", 'green'))
    if test:
       print(train.__len__(), valid.__len__(), test.__len__())
    else:
        print(train.__len__(), valid.__len__())
        print('No test file provided')
    
    #####Step3. call dataloader and create a list of minibatches
   
    # separate loader and minibatches for train, test, validation 
    # Note: mbs stands for minibatches
    print (' creating the list of training minibatches')
    if args.which_model == "RNN": pack_pad = True
    else: pack_pad = False
    train_mbs = list(tqdm(EHRdataloader(train, batch_size = args.batch_size, packPadMode = pack_pad, shuffle=False)))
    print (' creating the list of valid minibatches')
    valid_mbs = list(tqdm(EHRdataloader(valid, batch_size = args.batch_size, packPadMode = pack_pad, shuffle=False)))
    if test:
        print (' creating the list of test minibatches')
        test_mbs = list(tqdm(EHRdataloader(test, batch_size = args.batch_size, packPadMode = pack_pad, shuffle=False)))
    else:
        test_mbs = None
    
    #####Step2. Model loading
    if args.simple_fit == False and args.skip_model_fit == 0:
        print (args.which_model ,' model init')
        best_params = run_trials(train_mbs, valid_mbs, test_mbs, args, study_name)
        print(colored("\nOptimization done!", 'green'))
        print(timer.get_elapsed())
    else:
        best_params = {'embed_dim_exp': 7, 'hidden_size_exp': 7, 'dropout_r': 4, 'L2': 0.0009451300184944492, 'lr': 8, 'optimizer': 'adamax'}

    if args.skip_model_fit == 0:
        ehr_model = get_model(args, best_params)
        optimizer = get_obj(ehr_model, args, best_params)
        if use_cuda: 
            ehr_model = ehr_model.cuda()
        #######Step3. Train, validation and test. default: batch shuffle = true
        try:
            best_train_loss, best_val_loss, best_val_auc, best_test_auc, best_val_ep, best_model = ut.epochs_run(args.epochs, 
                          train = train_mbs, 
                          valid = valid_mbs, 
                          test = test_mbs, 
                          model = ehr_model, 
                          optimizer = optimizer,
                          shuffle = True, 
                          which_model = args.which_model, 
                          patience = 5,
                          model_out=out_dir + "models/" + args.lab_name + "_EHRmodel_" + get_date())
        #we can keyboard interupt now
        except KeyboardInterrupt:
            print(colored('-' * 89, 'green'))
            print(colored('Exiting from training early','green'))
    #save model & parameters
    
    #torch.save(best_model, + ".pth")
    #torch.save(best_model.state_dict(), out_dir + "models/" + args.lab_name + "_EHRmodel_" + get_date() + ".st")
    if args.date_model_fit == "": args.date_model_fit = get_date()
    best_model = torch.load(out_dir + "models/"  + args.lab_name + "_EHRmodel_" +args.date_model_fit+ ".pth", weights_only=False)
    best_model.load_state_dict(torch.load(out_dir + "models/"  + args.lab_name + "_EHRmodel_" + args.date_model_fit+ ".st"))
    best_model.eval()
    '''
    #later you can do to load previously trained model:
    best_model= torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    best_model.load_state_dict(torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.st'))
    best_model.eval()
    '''
    print(colored("\nFinal fitting done!", 'green'))
    print(timer.get_elapsed())

#    train_mbs = list(tqdm(EHRdataloader(train, batch_size = args.batch_size, packPadMode = pack_pad, shuffle=False)))
    out_data = get_out_data(best_model, train_mbs, valid_mbs, test_mbs, args.file_path_labels)
    out_data.to_csv(out_dir + out_name + "_preds_" + get_date() + ".csv", index=False)  
    crnt_report = create_report(best_model, out_data, display_scores=["logloss", "aucpr"], metric="logloss")
    pickle.dump(crnt_report, open(out_dir + out_name + "_report_" + get_date() + ".pkl", "wb"))  

    ## Report on all data

    fig = create_report_plot(out_data.loc[out_data.SET==1].TRUE_ABNORM, out_data.loc[out_data.SET==1].ABNORM_PROBS, out_data.loc[out_data.SET==1].ABNORM_PREDS)
    fig.savefig(out_dir + "plots/" + out_name + "_report_val_" + get_date() + ".png")   
    fig = create_report_plot(out_data.loc[out_data.SET==0].TRUE_ABNORM, out_data.loc[out_data.SET==0].ABNORM_PROBS, out_data.loc[out_data.SET==0].ABNORM_PREDS)
    fig.savefig(out_dir + "plots/" + out_name + "_report_train_" + get_date() + ".png")   
    fig = create_report_plot(out_data.loc[out_data.SET==1].TRUE_ABNORM, out_data.loc[out_data.SET==1].ABNORM_PROBS, out_data.loc[out_data.SET==1].ABNORM_PREDS, fg_down=True)
    fig.savefig(out_dir + "down/" + args.date_model_fit + "/" + out_name + "_" + study_name + "_report_val_" + get_date() + ".png")   
    fig.savefig(out_dir + "down/" + args.date_model_fit + "/" + out_name + "_" + study_name + "_report_val_" + get_date() + ".pdf")   
    fig = create_report_plot(out_data.loc[out_data.SET==0].TRUE_ABNORM, out_data.loc[out_data.SET==0].ABNORM_PROBS, out_data.loc[out_data.SET==0].ABNORM_PREDS, fg_down=True)
    fig.savefig(out_dir + "down/" + args.date_model_fit + "/" + out_name + "_" + study_name + "_report_train_" + get_date() + ".png")   
    fig.savefig(out_dir + "down/" + args.date_model_fit + "/" + out_name + "_" + study_name + "_report_train_" + get_date() + ".pdf")   
        
    eval_metrics, all_conf_mats = eval_subset(out_data, "ABNORM_PREDS", "ABNORM_PROBS", "TRUE_ABNORM", "TRUE_VALUE", out_dir, out_name + "_" + study_name, "all", args.n_boots, "bin")

    if args.save_csv == 1:
        eval_metrics.loc[eval_metrics.F1.notnull()].to_csv(out_dir + "down/" + args.date_model_fit + "/" + out_name + "_" + study_name +  "_evals_" + get_date() + ".csv", sep=",", index=False)
        all_conf_mats.to_csv(out_dir + out_name + "_confmats_" + get_date() + ".csv", sep=",", index=False)
    print(colored("\nEval and plotting done fitting done!", 'green'))
    print(timer.get_elapsed())
        
#do the main file functions and runs 
if __name__ == "__main__":
    main()