# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/"))
from utils import *
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

def set_metrics(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set):
    set_data = pred_data.loc[pred_data.SET == set]
    n_indv = set_data.shape[0]
    n_abnorm = set_data.loc[:,y_goal_col].sum()
    mse = skm.mean_squared_error(set_data.loc[:,y_cont_goal_col], set_data.loc[:,y_cont_pred_col])
    set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]

    f1 = skm.f1_score(set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col], average="macro")
    accuracy = skm.accuracy_score(set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col])
    accuracy_balanced = skm.balanced_accuracy_score(set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col])
    precision = skm.precision_score(set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col])
    recall = skm.recall_score(set_data.loc[:,y_goal_col], set_data.loc[:,y_pred_col])

    print("{} - MSE: {:.2f}   F1: {:.2f}   acurracy: {:.2f}".format(set_names[set], mse, f1, accuracy))
    eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": "All", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "MSE":mse, "F1":f1, "accuracy":accuracy, "accuracy_balanced":accuracy_balanced, "precision": precision, "recall": recall}, ignore_index=True)
    return(eval_metrics)

def conf_matrix_dfs(pred_data, y_goal_col, y_pred_col, all_conf_mats):
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=0)
    crnt_conf_mats = pd.DataFrame.from_dict({"ALL": cm[0].flatten()}, orient="index")
    crnt_conf_mats.loc[:,"SET"] = set_names[0]
    all_conf_mats = pd.concat([all_conf_mats, crnt_conf_mats])
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=1)
    crnt_conf_mats = pd.DataFrame.from_dict({"ALL": cm[0].flatten()}, orient="index")
    crnt_conf_mats.loc[:,"SET"] = set_names[1]
    all_conf_mats = pd.concat([all_conf_mats, crnt_conf_mats])
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=2)
    crnt_conf_mats = pd.DataFrame.from_dict({"ALL": cm[0].flatten()}, orient="index")
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

def time_evals(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, all_conf_mats):
    max_years = math.ceil(max(pred_data.TIME_LAST/365.25))
    time_scores = pd.DataFrame()
    conf_mats = pd.DataFrame()
    for set in [0,1,2]:
        set_data = pred_data.loc[pred_data.SET == set]
        set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]
        set_data = set_data.loc[set_data.loc[:,y_goal_col].notnull()]
        crnt_time_scores = pd.DataFrame({"SUBSET": pd.cut(pred_data.TIME_LAST, np.arange(0, 365.25*max_years, 365.25)).cat.categories})
        crnt_conf_mats = {}
        for level in pd.cut(set_data.TIME_LAST, np.arange(0, 365.25*11, 365.25)).cat.categories:
            crnt_data = set_data.loc[np.logical_and(set_data.TIME_LAST >= level.left, set_data.TIME_LAST <= level.right)]
            if crnt_data.shape[0] >= 5 and crnt_data.loc[:,y_goal_col].sum() >= 5:   
                conf_mats_square = conf_matrix(crnt_data, y_goal_col, y_pred_col, set, title="(" + str(level.left/365.25) + "," + str(level.right/365.25) + "]")
                crnt_conf_mats[level] = conf_mats_square[0].flatten()
                f1 = skm.f1_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col], average="macro")
                accuracy = skm.accuracy_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col])
                accuracy_balanced = skm.balanced_accuracy_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col])
                precision = skm.precision_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col])
                recall = skm.recall_score(crnt_data.loc[:,y_goal_col], crnt_data.loc[:,y_pred_col])

                mse = skm.mean_squared_error(crnt_data.loc[:,y_cont_goal_col], crnt_data.loc[:,y_cont_pred_col])
            
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "F1"] = f1
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "accuracy"] = accuracy
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "accuracy_balanced"] = accuracy_balanced
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "precision"] = precision
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "recall"] = recall

                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "MSE"] = mse
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "N_INDV"] = crnt_data.shape[0]
                crnt_time_scores.loc[crnt_time_scores.SUBSET == level, "N_ABNORM"] = crnt_data.loc[:,y_goal_col].sum()
                
        crnt_conf_mats = pd.DataFrame.from_dict(crnt_conf_mats, orient="index") 
        crnt_conf_mats.loc[:,"SET"] = set_names[set]
        crnt_time_scores.loc[:,"SET"] = set_names[set]
        if time_scores.shape[0] >= 5: 
            time_scores = pd.concat([time_scores, crnt_time_scores])
            conf_mats = pd.concat([conf_mats, crnt_conf_mats])
        else: 
            time_scores = crnt_time_scores.copy()
            conf_mats = crnt_conf_mats.copy()

    min_y = min([min(time_scores.F1), min(time_scores.accuracy), min(time_scores.precision), min(time_scores.recall)])

    fig, axes = plt.subplots(1, 3, figsize=(20,5))
    sns.lineplot(time_scores.loc[time_scores.SET=="Valid"][["SUBSET",  "F1", "accuracy", "accuracy_balanced"]].reset_index().melt(["index", "SUBSET"]), palette=["#D5694F", "#29557B", "#EAB034"],x="index", y="value", hue="variable", ax=axes[0])
    axes[0].set_ylim(min_y, 1)
    axes[0].set_xlabel("Years")
    axes[0].set_ylabel("")

    sns.lineplot(time_scores.loc[time_scores.SET=="Valid"][["SUBSET",  "F1", "precision", "recall"]].reset_index().melt(["index", "SUBSET"]), palette=["#D5694F", "#748AAA", "#CCB6AF"],x="index", y="value", hue="variable", ax=axes[1])
    axes[1].set_ylim(min_y, 1)
    axes[1].set_xlabel("Years")
    axes[1].set_label("")

    sns.lineplot(time_scores.loc[time_scores.SET=="Valid"][["SUBSET", "MSE"]].reset_index().melt(["index", "SUBSET"]), palette=["#841C26"], x="index", y="value", hue="variable", ax=axes[2])
    axes[2].set_xlabel("Years")
    axes[2].set_ylabel("MSE")
    
    eval_metrics = pd.concat([eval_metrics, time_scores])
    all_conf_mats = pd.concat([all_conf_mats, conf_mats])
    return(eval_metrics, all_conf_mats, fig)

def age_group_evals(pred_data,  y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics):
    age_scores = pd.DataFrame()
    age_groups = pred_data.AGE_GROUP.cat.remove_unused_categories().cat.categories

    for set in [0,1,2]:
        crnt_age_scores = pd.DataFrame({"SUBSET": age_groups})
        set_data = pred_data.loc[pred_data.SET == set]
        set_data = set_data.loc[set_data.loc[:,y_pred_col].notnull()]
        set_data = set_data.loc[set_data.loc[:,y_goal_col].notnull()]
        for group in age_groups:
            group_data = set_data.loc[set_data.AGE_GROUP == group]
            if group_data.shape[0] >= 5:
                f1 = skm.f1_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col], average="macro")
                accuracy = skm.accuracy_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col])
                mse = skm.mean_squared_error(group_data.loc[:,y_cont_goal_col], group_data.loc[:,y_cont_pred_col])
                precision = skm.precision_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col])
                recall = skm.recall_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col])
                accuracy_balanced = skm.balanced_accuracy_score(group_data.loc[:,y_goal_col], group_data.loc[:,y_pred_col])
                
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"F1"] = f1
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"accuracy"] = accuracy
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"MSE"] = mse
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"accuracy_balanced"] = accuracy_balanced
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"precision"] = precision
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"recall"] = recall

                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"N_INDV"] = group_data.shape[0]
                crnt_age_scores.loc[crnt_age_scores.SUBSET == group,"N_INDV"] = group_data.loc[:,y_goal_col].sum()


        crnt_age_scores.loc[:,"SET"] = set_names[set]
        if age_scores.shape[0] > 0: age_scores = pd.concat([age_scores, crnt_age_scores])
        else: age_scores = crnt_age_scores.copy()
            
    age_scores["AGE_MID"] = age_scores.SUBSET.transform(lambda x: x.mid)

    fig, axes = plt.subplots(1, 3, figsize=(20,5))
    min_y = min([min(age_scores.F1), min(age_scores.accuracy), min(age_scores.precision), min(age_scores.recall)])
    sns.lineplot(age_scores.loc[age_scores.SET=="Valid"][["AGE_MID",  "F1", "accuracy", "accuracy_balanced"]].melt(["AGE_MID"]), palette=["#D5694F", "#29557B", "#EAB034"],x="AGE_MID", y="value", hue="variable", ax=axes[0])
    axes[0].set_ylim(min_y, 1)
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("")

    sns.lineplot(age_scores.loc[age_scores.SET=="Valid"][["AGE_MID",  "F1", "precision", "recall"]].melt(["AGE_MID"]), palette=["#D5694F", "#748AAA", "#CCB6AF"],x="AGE_MID", y="value", hue="variable", ax=axes[1])
    axes[1].set_ylim(min_y, 1)
    axes[1].set_xlabel("Age")
    axes[1].set_label("")

    sns.lineplot(age_scores.loc[age_scores.SET=="Valid"][["AGE_MID", "MSE"]].melt(["AGE_MID"]), palette=["#841C26"], x="AGE_MID", y="value", hue="variable", ax=axes[2])
    axes[2].set_xlabel("Age")
    axes[2].set_ylabel("MSE")

    return(eval_metrics, fig)
    
def eval_preds(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, file_source, res_dir):
    sns.scatterplot(pred_data.loc[pred_data.SET == 1], x=y_cont_pred_col, y=y_cont_goal_col)

    all_conf_mats = pd.DataFrame()
    all_conf_mats = conf_matrix_dfs(pred_data, y_goal_col, y_pred_col, all_conf_mats)
    date = datetime.today().strftime("%Y-%m-%d")

    eval_metrics = pd.DataFrame(columns=("SET", "SUBSET", "N_INDV", "N_ABNORM", "MSE", "F1", "accuracy"))
    eval_metrics = set_metrics(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=0)
    eval_metrics = set_metrics(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=1)
    eval_metrics = set_metrics(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=2)
    eval_metrics, fig = age_group_evals(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics)
    
    fig.savefig(res_dir + "plots/agestrat_"+file_source+"_"+y_cont_goal_col+"_"+date+".png")
    fig.savefig(res_dir + "plots/agestrat_"+file_source+"_"+y_cont_goal_col+"_"+date+".pdf")
    eval_metrics, all_conf_mats, fig = time_evals(pred_data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, all_conf_mats)
    fig.savefig(res_dir + "plots/timestrat_"+file_source+"_"+y_cont_goal_col+"_"+date+".png")
    fig.savefig(res_dir + "plots/timestrat_"+file_source+"_"+y_cont_goal_col+"_"+date+".pdf")

    eval_metrics.loc[eval_metrics.MSE.notnull()].to_csv(res_dir + "tables/eval_"+file_source+y_cont_goal_col+date+".csv", sep=",", index=False)
    all_conf_mats.to_csv(res_dir + "tables/confmats_"+file_source+"_"+y_cont_goal_col+"_"+date+".csv", sep=",", index=False)

"""Data needs at least columns y_DATE and DATE. y_DATE is the date of the last observation, DATE is the date of the prediction."""
def merge_preds_and_labels(data, labels):
    data = pd.merge(data, labels, on="FINNGENID", how="inner")
    data.DATE = data.DATE.astype("datetime64[ns]")
    data.y_DATE = data.y_DATE.astype("datetime64[ns]")
    time_last = data.sort_values(["FINNGENID", "DATE"]).groupby("FINNGENID").tail(1).y_DATE - data.sort_values(["FINNGENID", "DATE"]).groupby("FINNGENID").tail(1).DATE 
    time_last = pd.merge(data[["FINNGENID"]], time_last.rename("TIME_LAST"), left_index=True, right_index=True, how="inner")
    time_last.TIME_LAST = time_last.TIME_LAST.dt.days
    data = pd.merge(data, time_last, on="FINNGENID", how="inner")
    return(data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/results/model_evals/carryover/")
    parser.add_argument("--file_path_preds", help="Full source file path to prediction file", required=True)
    parser.add_argument("--file_path_labels", help="Full source file path to labels file", required=True)
    parser.add_argument("--source_file", type=str, help="Name of orig file", required=True)
    parser.add_argument("--y_pred_col", type=str, help="Column name of prediction for ABNORMity", default="ABNORM_CUSTOM")
    parser.add_argument("--y_pred_cont_col", type=str, help="Column name of prediction for value", default="VALUE")
    parser.add_argument("--y_goal_col", type=str, help="Column name of goal for ABNORMity", default="y_ABNORM_CUSTOM")
    parser.add_argument("--y_goal_cont_col", type=str, help="Column name of goal for value", default="y_MEAN")
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

    init_logging(log_dir, log_file_name, date_time)

    #### Getting Data
    data = pd.read_csv(args.file_path_preds)
    labels = pd.read_csv(args.file_path_labels)
    data = merge_preds_and_labels(data, labels)
    data["AGE_GROUP"] = pd.cut(data["EVENT_AGE"], [18, 30, 40, 50, 60, 70, 80])

    #### Evaluating
    eval_preds(data, args.y_pred_col, args.y_pred_cont_col, args.y_goal_col, args.y_goal_cont_col, args.source_file, args.res_dir)
#python3 /home/ivm/valid/scripts/step6_eval.py --file_path_preds=/home/ivm/valid/data/processed_data/step5/preds_hba1c_2024-10-15_1-year_carryover_2024-10-16.csv --file_path_labels=/home/ivm/valid/data/processed_data/step4/labels_hba1c_2024-10-15_1-year_2024-10-16.csv --source_file=hba1c_2024-10-15 --y_goal_cont_col=y_MEAN

#python3 /home/ivm/valid/scripts/step3/eval_model.py --file_path_preds=/home/ivm/valid/data/processed_data/step3/preds_hba1c_ca_2024-10-11_1-year_carrylast_2024-10-14.csv --file_path_labels=labels_hba1c_ca_2024-10-11_1-year_2024-10-14.csv --source_file=hba1c_ca_2024-10-11_1-year_2024-10-14 


