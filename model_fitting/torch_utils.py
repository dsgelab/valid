
try:
    from termcolor import colored
except:
    def colored(string, color): return(string)
import sys

sys.path.append(("/home/ivm/valid/scripts/pytorch_ehr/"))


from EHRDataloader import EHRdataFromPickles, EHRdataloader
def read_files(root_dir: str,
                file_name_start: str,
                model_name: str):
    """Read files and create dataloaders."""

    train = EHRdataFromPickles(root_dir=root_dir, 
                               file=file_name_start+"_train.pkl", 
                               sort=True,
                               model=model_name)
    valid = EHRdataFromPickles(root_dir=root_dir, 
                               file=file_name_start+"_valid.pkl", 
                               sort=True,
                               model=model_name)
    test = EHRdataFromPickles(root_dir=root_dir, 
                              file=file_name_start+"_test.pkl", 
                              sort=True,
                              model=model_name)

    print(colored("\nSee an example data structure from training data:", 'green'))
    print(train.__getitem__(40, seeDescription = True))
    print(colored("\nSample data lengths for train, validation and test:", 'green'))
    if test:
       print(train.__len__(), valid.__len__(), test.__len__())
    else:
        print(train.__len__(), valid.__len__())
        print('No test file provided')

    return train, valid, test

from tqdm import tqdm
def create_mbs_from_dataloader(train, 
                               valid, 
                               test, 
                               model_name: str,
                               batch_size: int = 365,
                               min_batch_cases: int=0):
        # packPadMode is used to pack the padded sequences
    if model_name == "RNN": pack_pad = True
    else: pack_pad = False
    
    train_mbs = list(tqdm(EHRdataloader(train, 
                                        batch_size=batch_size, 
                                        packPadMode=pack_pad, 
                                        min_batch_cases=min_batch_cases,
                                        shuffle=True)))
    print (' creating the list of valid minibatches')
    valid_mbs = list(tqdm(EHRdataloader(valid, 
                                        batch_size=batch_size, 
                                        packPadMode=pack_pad, shuffle=False)))
    if test:
        print (' creating the list of test minibatches')
        test_mbs = list(tqdm(EHRdataloader(test, 
                                           batch_size=batch_size, 
                                           packPadMode=pack_pad, 
                                           shuffle=False)))
    else:
        test_mbs = None

    return train_mbs, valid_mbs, test_mbs

def get_mbs_from_files(root_dir: str,
                       file_name_start: str,
                       model_name: str,
                       batch_size: int = 365,
                        min_batch_cases: int=0):   
    """Read files and create dataloaders."""
    #####Step1. read files and create dataloaders
    train, valid, test = read_files(root_dir = root_dir,
                                    file_name_start = file_name_start,
                                    model_name = model_name)
    
    #####Step. call dataloader and create a list of minibatches
    print (' creating the list of training minibatches')
    train_mbs, valid_mbs, test_mbs = create_mbs_from_dataloader(train, 
                                                                valid, 
                                                                test, 
                                                                model_name,
                                                                batch_size,
                                                                min_batch_cases)
    
    return train_mbs, valid_mbs, test_mbs
    

from torch import optim
def get_torch_optimizer(ehr_model, 
                        eps, 
                        lr,
                        L2,
                        optimizer_name):
    #model optimizers to choose from.
    if optimizer_name == 'adam':
        optimizer = optim.Adam(ehr_model.parameters(), 
                               lr=1/(10*lr), 
                               weight_decay=L2,
                               eps=eps)
    elif optimizer_name  == 'adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), 
                                   lr=1/(10*lr), 
                                   weight_decay=L2,
                                   eps=eps)
    elif optimizer_name  == 'adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), 
                                  lr=1/(10*lr), 
                                  weight_decay=L2) 
    elif optimizer_name  == 'adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), 
                                 lr=1/(10*lr), 
                                 weight_decay=L2,
                                 eps=eps)
    elif optimizer_name == 'asgd':
        optimizer = optim.ASGD(ehr_model.parameters(), 
                               lr=1/(10*lr), 
                               weight_decay=L2)
    elif optimizer_name  == 'rmsprop':
        optimizer = optim.RMSprop(ehr_model.parameters(), 
                                  lr=1/(10*lr), 
                                  weight_decay=L2,
                                  eps=eps)
    elif optimizer_name  == 'rprop':
        optimizer = optim.Rprop(ehr_model.parameters(), lr=1/(10*lr))
    elif optimizer_name  == 'sgd':
        optimizer = optim.SGD(ehr_model.parameters(), 
                              lr=1/(10*lr), 
                              weight_decay=L2)
    else:
        raise NotImplementedError
    return(optimizer)

from models import EHR_RNN, EHR_TLSTM, EHR_LR_emb, RETAIN
def get_model(model_name, 
              embed_dim_exp,
              hidden_size_exp,
              n_layers,
              dropout_r,
              cell_type,
              bii,
              time,
              preTrainEmb,
              input_size,
              final_embed_dim_exp):
    """Get the model."""
    if model_name == 'RNN': 
        ehr_model = EHR_RNN(input_size=input_size, 
                            embed_dim=2**embed_dim_exp, 
                            hidden_size=2**hidden_size_exp,
                            n_layers=n_layers,
                            dropout_r=0.1*dropout_r,
                            cell_type=cell_type,
                            bii=bii,
                            time=time,
                            preTrainEmb=preTrainEmb) 
    elif model_name == 'TLSTM': 
        ehr_model = EHR_TLSTM(input_size=input_size, 
                              embed_dim=2**embed_dim_exp, 
                              hidden_size=2**hidden_size_exp,
                              n_layers=1,
                              dropout_r=0.1*dropout_r,
                              cell_type= 'TLSTM', #doesn't support normal cell types
                              bii=False, 
                              time=time, 
                              preTrainEmb=preTrainEmb,
                             final_embed_dim_exp=final_embed_dim_exp)  
    elif model_name == 'RETAIN': 
        ehr_model = RETAIN(embed_dim=2**embed_dim_exp, 
                           hidden_size=2**hidden_size_exp,
                           n_layers=1,
                           input_size=input_size)
    else: 
        ehr_model = EHR_LR_emb(input_size=input_size,
                               embed_dim=2**embed_dim_exp, 
                               preTrainEmb=preTrainEmb)

    return(ehr_model)

from utils_final import get_preds
import polars as pl
import logging
from sklearn.metrics import precision_recall_curve
import numpy as np
def get_crnt_preds(data_mbs, 
                   mdl, 
                   crnt_set=0):
    t2_fids, t2_labels, t2_scores = get_preds(mdl, data_mbs)
    crnt_preds = pl.DataFrame({"FINNGENID": t2_fids, 
                               "TRUE_ABNORM": t2_labels, 
                               "ABNORM_PROBS": t2_scores})
    precision_, recall_, proba = precision_recall_curve(crnt_preds["TRUE_ABNORM"], 
                                                        crnt_preds["ABNORM_PROBS"])
    optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
    logging.info(f"Optimal cut-off set {crnt_set} for prediction based on PR {optimal_proba_cutoff}")
    crnt_preds = crnt_preds.with_columns((crnt_preds["ABNORM_PROBS"]>optimal_proba_cutoff).alias("ABNORM_PREDS").cast(pl.Int64))
    
    return(crnt_preds)
    
from general_utils import read_file
from model_fit_utils import get_cont_goal_col_name
import polars as pl
def get_out_data(model_final, 
                 train_mbs, 
                 valid_mbs, 
                 test_mbs, 
                 file_path_labels,
                 goal):
    ###### Predictions
    train_preds = get_crnt_preds(train_mbs, model_final, 0)
    test_preds = get_crnt_preds(test_mbs, model_final, 2)
    val_preds = get_crnt_preds(valid_mbs, model_final, 1)
    all_preds = pl.concat([train_preds, test_preds, val_preds])

    ####### Original data
    labels = read_file(file_path_labels)
    if "y_MEAN_ABNORM" in labels.columns: labels = labels.with_columns(pl.when(pl.col.y_MEAN_ABNORM==0).then(0).otherwise(1).alias("y_MEAN_ABNORM"))
    if "y_NEXT_ABNORM" in labels.columns: labels = labels.with_columns(pl.when(pl.col.y_NEXT_ABNORM==0).then(0).otherwise(1).alias("y_NEXT_ABNORM"))
    if "y_MIN_ABNORM" in labels.columns: labels = labels.with_columns(pl.when(pl.col.y_MIN_ABNORM==0).then(0).otherwise(1).alias("y_MIN_ABNORM"))
    if "y_MAX_ABNORM" in labels.columns: labels = labels.with_columns(pl.when(pl.col.y_MAX_ABNORM==0).then(0).otherwise(1).alias("y_MAX_ABNORM"))

    labels = labels.with_columns(pl.col(get_cont_goal_col_name(goal, labels.columns)).alias("TRUE_VALUE"))
    labels = labels.with_columns(pl.col(goal).alias("TRUE_ABNORM"))
    labels = labels.join(all_preds, on="FINNGENID", how="inner")
    return(labels)