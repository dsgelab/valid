# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from utils import *
from plot_utils import *
# Standard stuff
import numpy as np
import pandas as pd
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime
# Metrics and math
import sklearn.metrics as skm
import math

set_names = {1:"Valid", 2: "Test", 0: "Train"}
sns.set_style("whitegrid")

"""Bootstrapping metrics by shuffling observations and predictions through sampling with redrawing."""
def bootstrap_metric(func, obs, preds, n_boots=500, rng_seed=42):
    bootstraps = []
    obs = np.array(obs)
    preds = np.array(preds)
    rng = np.random.RandomState(rng_seed)
    try:
        total_est = func(obs, preds)
    except:
        return(np.nan, np.nan, np.nan)
    for i in np.arange(n_boots):
        idxs = rng.randint(0, len(obs), len(obs))
        try: # tweedie sometimes has issues
            bootstraps.append(func(obs[idxs], preds[idxs]))
        except:
            continue
    if len(bootstraps) > 0:
        # Sorting
        bootstraps = np.array(bootstraps)
        bootstraps.sort()
        # 95% CI
        ci_low = bootstraps[int(0.025*len(bootstraps))]
        ci_high = bootstraps[int(0.975*len(bootstraps))]

        return(total_est, ci_low, ci_high)
    else:
        return(np.nan, np.nan, np.nan)
        
"""Calculate metrics on given set of data."""
def set_metrics(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set, subset_name="all", n_boots=500, train_type="cont"):
    set_data = pred_data.loc[pred_data.SET == set]
    n_indv = set_data.shape[0]
    n_abnorm = set_data.loc[:,y_goal_col].sum()

    if train_type != "bin":
        # Measurement for continuous value
        mse = bootstrap_metric(skm.mean_squared_error, set_data.loc[:,y_cont_goal_col], set_data.loc[:,y_cont_pred_col], n_boots=n_boots)
        tweedie = bootstrap_metric(lambda x, y: skm.d2_tweedie_score(x, y, power=3), set_data.loc[:,y_cont_goal_col], set_data.loc[:,y_cont_pred_col], n_boots=n_boots)
    else:
        AUC = bootstrap_metric(skm.roc_auc_score, set_data.loc[:,y_goal_col], set_data.loc[:,y_cont_pred_col], n_boots=n_boots)
        avPRC = bootstrap_metric(skm.average_precision_score, set_data.loc[:,y_goal_col], set_data.loc[:,y_cont_pred_col], n_boots=n_boots)

    # Measurement for binary outcome
    set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]
    set_data = set_data.loc[set_data.loc[:,y_goal_col].notnull()]

    ## Making sure at least one abnormal is present, otherwise metrics make no sense
    if n_abnorm > 0:
        accuracy = bootstrap_metric(skm.accuracy_score, set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col], n_boots=n_boots)
        f1 = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col], n_boots=n_boots)
        precision = bootstrap_metric(lambda x, y: skm.precision_score(x, y, zero_division=0), set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col], n_boots=n_boots)
        recall = bootstrap_metric(lambda x, y: skm.recall_score(x, y, zero_division=0), set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col], n_boots=n_boots)

        # Results
        if train_type!="bin":
            print("{} - Tweedie: {:.2f} ({:.2f}-{:.2f})  MSE: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})   acurracy: {:.2f} ({:.2f}-{:.2f})   precision {:.2f} ({:.2f}-{:.2f})    recall {:.2f} ({:.2f}-{:.2f})".format(set_names[set], tweedie[0], tweedie[1], tweedie[2], mse[0], mse[1], mse[2], f1[0], f1[1], f1[2], accuracy[0], accuracy[1], accuracy[2], precision[0], precision[1], precision[2], recall[0], recall[1], recall[2]))
        else:
            print("{} - AUC: {:.2f} ({:.2f}-{:.2f})  avPRC: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})   acurracy: {:.2f} ({:.2f}-{:.2f})   precision {:.2f} ({:.2f}-{:.2f})    recall {:.2f} ({:.2f}-{:.2f})".format(set_names[set], AUC[0], AUC[1], AUC[2], avPRC[0], avPRC[1], avPRC[2], f1[0], f1[1], f1[2], accuracy[0], accuracy[1], accuracy[2], precision[0], precision[1], precision[2], recall[0], recall[1], recall[2]))

        ## First row in dataset
        if eval_metrics.shape[0] == 0:
            if train_type!="bin":
                eval_metrics = pd.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": [n_indv], "N_ABNORM": [n_abnorm], "tweedie": [tweedie[0]], "tweedie_CIneg": [tweedie[1]], "tweedie_CIpos": [tweedie[2]], "MSE": [mse[0]], "MSE_CIneg": [mse[1]], "MSE_CIpos": [mse[2]], "F1":[f1[0]], "F1_CIneg": [f1[1]], "F1_CIpos": [f1[2]], "accuracy": [accuracy[0]], "accuracy_CIneg": [accuracy[1]], "accuracy_CIpos": [accuracy[2]], "precision": [precision[0]], "precision_CIneg": [precision[1]], "precision_CIpos": [precision[2]], "recall": [recall[0]], "recall_CIneg": [recall[1]], "recall_CIpos": [recall[2]]})
            else:
                eval_metrics = pd.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": [n_indv], "N_ABNORM": [n_abnorm], "AUC": [AUC[0]], "AUC_CIneg": [AUC[1]], "AUC_CIpos": [AUC[2]], "avPRC": [avPRC[0]], "avPRC_CIneg": [avPRC[1]], "avPRC_CIpos": [avPRC[2]], "F1":[f1[0]], "F1_CIneg": [f1[1]], "F1_CIpos": [f1[2]], "accuracy": [accuracy[0]], "accuracy_CIneg": [accuracy[1]], "accuracy_CIpos": [accuracy[2]], "precision": [precision[0]], "precision_CIneg": [precision[1]], "precision_CIpos": [precision[2]], "recall": [recall[0]], "recall_CIneg": [recall[1]], "recall_CIpos": [recall[2]]})

        ## All other rows
        else:
            if train_type!="bin":
                eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "tweedie": tweedie[0], "tweedie_CIneg": tweedie[1], "tweedie_CIpos": tweedie[2], "MSE":mse[0], "MSE_CIneg": mse[1], "MSE_CIpos": mse[2], "F1":f1[0], "F1_CIneg": f1[1], "F1_CIpos": f1[2], "accuracy":accuracy[0], "accuracy_CIneg":accuracy[1], "accuracy_CIpos": accuracy[2], "precision": precision[0], "precision_CIneg": precision[1], "precision_CIpos": precision[2], "recall": recall[0], "recall_CIneg": recall[1], "recall_CIpos": recall[2]}, ignore_index=True)
            else:
                eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "AUC": AUC[0], "AUC_CIneg": AUC[1], "AUC": AUC[2], "avPRC":avPRC[0], "avPRC_CIneg": avPRC[1], "avPRC_CIpos": avPRC[2], "F1":f1[0], "F1_CIneg": f1[1], "F1_CIpos": f1[2], "accuracy":accuracy[0], "accuracy_CIneg":accuracy[1], "accuracy_CIpos": accuracy[2], "precision": precision[0], "precision_CIneg": precision[1], "precision_CIpos": precision[2], "recall": recall[0], "recall_CIneg": recall[1], "recall_CIpos": recall[2]}, ignore_index=True)

    ## Only continuous metrics
    else:
        # Results
        if train_type!="bin":
            print("{} - Tweedie: {:.2f} ({:.2f}-{:.2f})  MSE: {:.2f} ({:.2f}-{:.2f})".format(set_names[set], tweedie[0], tweedie[1], tweedie[2], mse[0], mse[1], mse[2]))
            eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name,"GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "tweedie": tweedie[0], "tweedie_CIneg": tweedie[1], "tweedie_CIpos": tweedie[2], "MSE":mse[0], "MSE_CIneg": mse[1], "MSE_CIpos": mse[2]}, ignore_index=True)
        else:
            print("{} - AUC: {:.2f} ({:.2f}-{:.2f})  avPRC: {:.2f} ({:.2f}-{:.2f}) ".format(set_names[set], AUC[0], AUC[1], AUC[2], avPRC[0], avPRC[1], avPRC[2]))
            eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "AUC": AUC[0], "AUC_CIneg": AUC[1], "AUC": AUC_CIpos[2], "avPRC":avPRC[0], "avPRC_CIneg": avPRC[1], "avPRC_CIpos": avPRC[2]}, ignore_index=True)

    return(eval_metrics)

def conf_matrix_dfs(pred_data, y_goal_col, y_pred_col, all_conf_mats):
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=0)
    crnt_conf_mats = pd.DataFrame.from_dict({"all": cm[0].flatten()}, orient="index")
    crnt_conf_mats.loc[:,"SET"] = set_names[0]
    all_conf_mats = pd.concat([all_conf_mats, crnt_conf_mats])
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=1)
    crnt_conf_mats = pd.DataFrame.from_dict({"all": cm[0].flatten()}, orient="index")
    crnt_conf_mats.loc[:,"SET"] = set_names[1]
    all_conf_mats = pd.concat([all_conf_mats, crnt_conf_mats])
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=2)
    crnt_conf_mats = pd.DataFrame.from_dict({"all": cm[0].flatten()}, orient="index")
    crnt_conf_mats.loc[:,"SET"] = set_names[2]
    all_conf_mats = pd.concat([all_conf_mats, crnt_conf_mats])
    
    return(all_conf_mats)

def conf_matrix(pred_data, y_goal_col, y_pred_col, set=1, title=""):
    set_data = pred_data.loc[pred_data.SET == set]
    set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]
    set_data = set_data.loc[set_data.loc[:,y_goal_col].notnull()]
    cm_norm = skm.confusion_matrix(set_data.loc[:,y_goal_col].values, set_data.loc[:,y_pred_col].values, normalize="true")
    cm = skm.confusion_matrix(set_data.loc[:,y_goal_col].values, set_data.loc[:,y_pred_col].values)
    return(cm, cm_norm)

def time_evals(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, all_conf_mats, subset_name):
    max_years = math.ceil(max(pred_data.TIME_LAST/365.25))
    time_scores = pd.DataFrame()
    conf_mats = pd.DataFrame()
    for set in [0,1,2]:
        set_data = pred_data.loc[pred_data.SET == set]
        set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]
        set_data = set_data.loc[set_data.loc[:,y_goal_col].notnull()]
        crnt_time_scores = pd.DataFrame({"GROUP": pd.cut(pred_data.TIME_LAST, np.arange(0, 365.25*max_years, 365.25)).cat.categories})
        crnt_conf_mats = {}
        for level in pd.cut(set_data.TIME_LAST, np.arange(0, 365.25*11, 365.25)).cat.categories:
            crnt_data = set_data.loc[np.logical_and(set_data.TIME_LAST >= level.left, set_data.TIME_LAST <= level.right)]
            if crnt_data.shape[0] >= 5 and crnt_data.loc[:,y_goal_col].sum() >= 5:   
                conf_mats_square = conf_matrix(crnt_data, y_goal_col, y_pred_col, set, title="(" + str(level.left/365.25) + "," + str(level.right/365.25) + "]")
                crnt_conf_mats[level] = conf_mats_square[0].flatten()

                mse = skm.mean_squared_error(crnt_data.loc[:,y_cont_goal_col], crnt_data.loc[:,y_cont_pred_col])
                try:
                    tweedie = skm.d2_tweedie_score(crnt_data.loc[:,y_cont_goal_col], crnt_data.loc[:,y_cont_pred_col], power=3)
                    if tweedie < 0: tweedie = np.nan
                except:
                    tweedie = np.nan
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "tweedie"] = tweedie
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "MSE"] = mse
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "N_INDV"] = crnt_data.shape[0]
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "N_ABNORM"] = crnt_data.loc[:,y_goal_col].sum()

                f1 = skm.f1_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col], average="macro", zero_division=0)
                accuracy = skm.accuracy_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col])
                precision = skm.precision_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col], zero_division=0)
                recall = skm.recall_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col], zero_division=0)

                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "F1"] = f1
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "accuracy"] = accuracy
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "precision"] = precision
                crnt_time_scores.loc[crnt_time_scores.GROUP == level, "recall"] = recall
                
        crnt_conf_mats = pd.DataFrame.from_dict(crnt_conf_mats, orient="index") 
        crnt_conf_mats.loc[:,"SET"] = set_names[set]
        crnt_time_scores.loc[:,"SET"] = set_names[set]
        crnt_conf_mats.loc[:,"SUBSET"] = subset_name
        crnt_time_scores.loc[:,"SUBSET"] = subset_name
        
        if time_scores.shape[0] >= 5: 
            time_scores = pd.concat([time_scores, crnt_time_scores])
            conf_mats = pd.concat([conf_mats, crnt_conf_mats])
        else: 
            time_scores = crnt_time_scores.copy()
            conf_mats = crnt_conf_mats.copy()
    
    fig = plot_scores_metrics(time_scores, eval_metrics, "Years", "GROUP", subset_name)

    eval_metrics = pd.concat([eval_metrics, time_scores])
    all_conf_mats = pd.concat([all_conf_mats, conf_mats])
    return(eval_metrics, all_conf_mats, fig)

def age_group_evals(pred_data,  y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, subset_name):
    age_scores = pd.DataFrame()
    age_groups = pred_data.AGE_GROUP.cat.remove_unused_categories().cat.categories

    for set in [0,1,2]:
        crnt_age_scores = pd.DataFrame({"GROUP": age_groups})
        set_data = pred_data.loc[pred_data.SET == set]
        set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]
        set_data = set_data.loc[set_data.loc[:,y_goal_col].notnull()]
        for group in age_groups:
            group_data = set_data.loc[set_data.AGE_GROUP == group]
            mse = skm.mean_squared_error(group_data.loc[:,y_cont_goal_col], group_data.loc[:,y_cont_pred_col])
            try:
                tweedie = skm.d2_tweedie_score(group_data.loc[:,y_cont_goal_col], group_data.loc[:,y_cont_pred_col], power=3)
                if tweedie < 0: tweedie = np.nan
            except:
                tweedie = np.nan
            
            crnt_age_scores.loc[crnt_age_scores.GROUP == group,"tweedie"] = tweedie
            crnt_age_scores.loc[crnt_age_scores.GROUP == group,"MSE"] = mse
            crnt_age_scores.loc[crnt_age_scores.GROUP == group,"N_INDV"] = group_data.shape[0]
            crnt_age_scores.loc[crnt_age_scores.GROUP == group,"N_ABNORM"] = group_data.loc[:,y_goal_col].sum()
            if group_data.shape[0] >= 5:
                f1 = skm.f1_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col], average="macro", zero_division=0)
                accuracy = skm.accuracy_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col])
                precision = skm.precision_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col], zero_division=0)
                recall = skm.recall_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col], zero_division=0)
                    
                crnt_age_scores.loc[crnt_age_scores.GROUP == group,"F1"] = f1
                crnt_age_scores.loc[crnt_age_scores.GROUP == group,"accuracy"] = accuracy
                crnt_age_scores.loc[crnt_age_scores.GROUP == group,"precision"] = precision
                crnt_age_scores.loc[crnt_age_scores.GROUP == group,"recall"] = recall

        crnt_age_scores.loc[:,"SET"] = set_names[set]
        crnt_age_scores.loc[:,"SUBSET"] = subset_name
        if age_scores.shape[0] > 0: age_scores = pd.concat([age_scores, crnt_age_scores])
        else: age_scores = crnt_age_scores.copy()
    eval_metrics = pd.concat([eval_metrics, age_scores])
    age_scores["AGE_MID"] = age_scores.GROUP.transform(lambda x: x.mid)
    fig = plot_scores_metrics(age_scores, eval_metrics, "Age", "AGE_MID", subset_name)
    
    return(eval_metrics, fig)
    
def eval_subset(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, fig_path, table_path, subset_name="all", n_boots=500, train_type="cont"):
    eval_metrics = pd.DataFrame()
    all_conf_mats = pd.DataFrame()

    if train_type != "binary":
        N_indv = data.shape[0]
        g, newcol_x, newcol_y = plot_observerd_vs_predicted_min5(data, y_cont_goal_col, y_cont_pred_col)
        g.savefig(fig_path + "_" + subset_name + "_scatter_min5.png", dpi=300)
        g.savefig(fig_path + "_" + subset_name + "_scatter_min5.pdf")
        newcol_x.value_counts().to_csv(table_path + "/extra_counts/_" + subset_name + "_indv_counts_x.csv", sep=",")
        newcol_y.value_counts().to_csv(table_path + "/extra_counts/_" + subset_name + "_indv_counts_y.csv", sep=",")
        
        g = plot_observerd_vs_predicted(data, y_cont_goal_col, y_cont_pred_col, y_goal_col)
        g.savefig(fig_path + "_" + subset_name + "_scatter.png", dpi=300)
    else:
        g, newcol_x, newcol_y = plot_observed_vs_probability_min5(data, y_cont_goal_col, y_cont_pred_col)
        g.savefig(fig_path + "_" + subset_name + "_scatter_min5.png", dpi=300)
        g.savefig(fig_path + "_" + subset_name + "_scatter_min5.pdf")
        newcol_x.value_counts().to_csv(table_path + "_" + subset_name + "_indv_counts_x.csv", sep=",")
        newcol_y.value_counts().to_csv(table_path + "_" + subset_name + "_indv_counts_y.csv", sep=",")

    all_conf_mats = conf_matrix_dfs(data, y_goal_col, y_pred_col, all_conf_mats)
    # Metrics with bootstrap for different sets
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=0, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=1, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=2, subset_name=subset_name, n_boots=n_boots, train_type=train_type)

    if train_type == "cont":
    ## Age-group level
        eval_metrics, fig = age_group_evals(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, subset_name)
        fig.savefig(fig_path + "_" + subset_name + "_agestrata.png", dpi=300)
        fig.savefig(fig_path + "_" + subset_name + "_agestrata.pdf")
    
        ## Time level
        if "TIME_LAST" in data.columns:
            eval_metrics, all_conf_mats, fig = time_evals(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, all_conf_mats, subset_name)
            fig.savefig(fig_path + "_" + subset_name +"_timestrata.png", dpi=300)
            fig.savefig(fig_path + "_" + subset_name + "_timestrata.pdf")
    
    return(eval_metrics, all_conf_mats)
    
def eval_preds(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, fig_path, table_path, n_boots=500, train_type="cont"):
    eval_metrics = pd.DataFrame()
    all_conf_mats = pd.DataFrame()

    eval_metrics, all_conf_mats = eval_subset(eval_metrics, all_conf_mats, pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, n_boots=n_boots, train_type=train_type)
    # Data diag
    print("No data diag-------------------------------------------------------------------------------------------------------------------------------")
    subset_data = data.loc[np.logical_or(data.DATA_DIAG_CUSTOM_DATE.isnull(), data.DATA_DIAG_CUSTOM_DATE>=data.START_DATE)]
    eval_metrics, all_conf_mats = eval_subset(eval_metrics, all_conf_mats, subset_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, subset_name="nodatadiag", n_boots=n_boots, train_type=train_type)

    # No data abnormal
    print("No abnormal-------------------------------------------------------------------------------------------------------------------------------")
    subset_data = data.loc[np.logical_or(data.FIRST_ABNORM_CUSTOM_DATE.isnull(), data.FIRST_ABNORM_CUSTOM_DATE>=data.START_DATE)]
    eval_metrics, all_conf_mats = eval_subset(eval_metrics, all_conf_mats, subset_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, subset_name="noabnorm", n_boots=n_boots, train_type=train_type)

    # No fg abnormal
    print("No fg abnormal-------------------------------------------------------------------------------------------------------------------------------")
    subset_data = data.loc[np.logical_or(data.FIRST_ABNORM_DATE.isnull(), data.FIRST_ABNORM_DATE>=data.START_DATE)]
    eval_metrics, all_conf_mats = eval_subset(eval_metrics, all_conf_mats, subset_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, subset_name="nofgabnorm", n_boots=n_boots, train_type=train_type)
    
    return(eval_metrics, all_conf_mats)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/results/model_evals/carryover/")
    parser.add_argument("--file_path_preds", help="Full source file path to prediction file", required=True)
    parser.add_argument("--file_path_labels", help="Full source file path to labels file", required=True)
    parser.add_argument("--file_path_meta", help="Full source file path to labels file", required=True)
    parser.add_argument("--file_path_diags", help="Full source file path to labels file", required=True)
    parser.add_argument("--source_file", type=str, help="Name of orig file", required=True)
    parser.add_argument("--y_pred_col", type=str, help="Column name of prediction for abnormality", default="ABNORM_CUSTOM")
    parser.add_argument("--y_pred_cont_col", type=str, help="Column name of prediction for value", default="VALUE")
    parser.add_argument("--y_goal_col", type=str, help="Column name of goal for abnormality", default="y_ABNORM_CUSTOM")
    parser.add_argument("--y_goal_cont_col", type=str, help="Column name of goal for value", default="y_MEAN")
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics", default=500)
    parser.add_argument("--train_type", type=str, help="Type of training, i.e. regression will be cont and logistic regression binary.", default="cont")
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)

    args = parser.parse_args()

    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
     
if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = args.source_file + "_eval_" + date
    log_file_name = args.source_file + "_eval_" + date_time
    make_dir(args.res_dir)
    make_dir(log_dir)
    make_dir(args.res_dir+"plots/")
    make_dir(args.res_dir+"tables/")
    fig_path = args.res_dir + "plots/"+args.source_file+"_"+date
    table_path = args.res_dir + "tables/"+args.source_file+"_"+date

    init_logging(log_dir, log_file_name, date_time)
    
    #### Getting Data
    data = pd.read_csv(args.file_path_preds)
    labels = pd.read_csv(args.file_path_labels)
    data = merge_preds_and_labels(data, labels)
    # Diag and abnorm for subsetting
    diags = pd.read_csv(args.file_path_diags)
    abnorms = pd.read_csv(args.file_path_meta)[["FINNGENID", "DIAG_DATE", "FIRST_ABNORM_DATE", "FIRST_ABNORM_CUSTOM_DATE"]]
    data = pd.merge(data, diags, on="FINNGENID", how="left")
    data = pd.merge(data, abnorms, on="FINNGENID", how="left")
    data.DIAG_DATE = data.DIAG_DATE.astype("datetime64[ns]")
    data.FIRST_ABNORM_DATE = data.FIRST_ABNORM_DATE.astype("datetime64[ns]")
    data.FIRST_ABNORM_CUSTOM_DATE = data.FIRST_ABNORM_CUSTOM_DATE.astype("datetime64[ns]")
    data.DATA_DIAG_DATE = data.DATA_DIAG_DATE.astype("datetime64[ns]")
    data.DATA_DIAG_CUSTOM_DATE = data.DATA_DIAG_CUSTOM_DATE.astype("datetime64[ns]")
    # age grouping
    data["AGE_GROUP"] = pd.cut(data["EVENT_AGE"], [18, 30, 40, 50, 60, 70, 80])
    
    #### Evaluating
    eval_metrics, all_conf_mats = eval_preds(data, args.y_pred_col, args.y_pred_cont_col, args.y_goal_col, args.y_goal_cont_col, fig_path, table_path, args.n_boots, args.train_type)
    if args.save_csv == 1:
        eval_metrics.loc[eval_metrics.F1.notnull()].to_csv(table_path + "_evals.csv", sep=",", index=False)
        all_conf_mats.to_csv(table_path + "_confmats.csv", sep=",", index=False)
    print(timer.get_elapsed())
    logger.info("Time total: "+timer.get_elapsed())
