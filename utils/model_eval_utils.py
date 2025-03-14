import matplotlib
matplotlib.use("Agg")
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
import matplotlib.pyplot as plt
import pandas as pd
from plot_utils import *
from utils import *
import sklearn.metrics as skm

def model_memory_size(clf):
    try:
        return sys.getsizeof(pickle.dumps(clf))
    except:
        return 0
    
def get_train_type(metric):
    if metric == "tweedie": return("cont")
    if metric == "mse": return("cont")
    if metric == "logloss": return("bin")
    if metric == "auc": return("bin")
    if metric == "F1" or metric == "f1" or metric == "aucpr": return("bin")

def get_score_func_based_on_metric(metric):
    if metric == "tweedie": return(skm.d2_tweedie_score)
    if metric == "mse": return(skm.mean_squared_error)
    if metric == "logloss": return(skm.log_loss)
    if metric == "auc": return(skm.auc)
    if metric == "F1" or metric == "f1": return(skm.f1_score)
    if metric == "aucpr" or metric == "AUPRC" or metric == "auprc": return(skm.average_precision_score)

set_names = {1:"Valid", 2: "Test", 0: "Train"}

def bootstrap_metric(func, obs, preds, n_boots=500, rng_seed=42):
    """Bootstrapping metrics by shuffling observations and predictions through sampling with redrawing."""
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
    n_abnorm = set_data[y_goal_col].sum()

    if train_type != "bin":
        # Measurement for continuous value
        mse = bootstrap_metric(skm.mean_squared_error, set_data[y_cont_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)
        tweedie = bootstrap_metric(lambda x, y: skm.d2_tweedie_score(x, y, power=3), set_data[y_cont_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)
    else:
        AUC = bootstrap_metric(skm.roc_auc_score, set_data[y_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)
        avPRC = bootstrap_metric(skm.average_precision_score, set_data[y_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)

    ## Making sure at least one abnormal is present, otherwise metrics make no sense
    if n_abnorm > 0:
        accuracy = bootstrap_metric(skm.accuracy_score, set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)
        f1 = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)
        precision = bootstrap_metric(lambda x, y: skm.precision_score(x, y, zero_division=0), set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)
        recall = bootstrap_metric(lambda x, y: skm.recall_score(x, y, zero_division=0), set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)

        # Results
        if train_type!="bin":
            print("{} - Tweedie: {:.2f} ({:.2f}-{:.2f})  MSE: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})   acurracy: {:.2f} ({:.2f}-{:.2f})   precision {:.2f} ({:.2f}-{:.2f})    recall {:.2f} ({:.2f}-{:.2f})".format(set_names[set], tweedie[0], tweedie[1], tweedie[2], mse[0], mse[1], mse[2], f1[0], f1[1], f1[2], accuracy[0], accuracy[1], accuracy[2], precision[0], precision[1], precision[2], recall[0], recall[1], recall[2]))
        else:
            print("{} - AUC: {:.2f} ({:.2f}-{:.2f})  avPRC: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})   acurracy: {:.2f} ({:.2f}-{:.2f})   precision {:.2f} ({:.2f}-{:.2f})    recall {:.2f} ({:.2f}-{:.2f})".format(set_names[set], AUC[0], AUC[1], AUC[2], avPRC[0], avPRC[1], avPRC[2], f1[0], f1[1], f1[2], accuracy[0], accuracy[1], accuracy[2], precision[0], precision[1], precision[2], recall[0], recall[1], recall[2]))

        ## First row in dataset
        if eval_metrics.shape[0] == 0:
            if train_type!="bin":
                eval_metrics = pd.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all","N_INDV": [n_indv], "N_ABNORM": [n_abnorm], "tweedie": [tweedie[0]], "tweedie_CIneg": [tweedie[1]], "tweedie_CIpos": [tweedie[2]], "MSE": [mse[0]], "MSE_CIneg": [mse[1]], "MSE_CIpos": [mse[2]], "F1":[f1[0]], "F1_CIneg": [f1[1]], "F1_CIpos": [f1[2]], "accuracy": [accuracy[0]], "accuracy_CIneg": [accuracy[1]], "accuracy_CIpos": [accuracy[2]], "precision": [precision[0]], "precision_CIneg": [precision[1]], "precision_CIpos": [precision[2]], "recall": [recall[0]], "recall_CIneg": [recall[1]], "recall_CIpos": [recall[2]]})
            else:
                eval_metrics = pd.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": [n_indv], "N_ABNORM": [n_abnorm], "AUC": [AUC[0]], "AUC_CIneg": [AUC[1]], "AUC_CIpos": [AUC[2]], "avPRC": [avPRC[0]], "avPRC_CIneg": [avPRC[1]], "avPRC_CIpos": [avPRC[2]], "F1":[f1[0]], "F1_CIneg": [f1[1]], "F1_CIpos": [f1[2]], "accuracy": [accuracy[0]], "accuracy_CIneg": [accuracy[1]], "accuracy_CIpos": [accuracy[2]], "precision": [precision[0]], "precision_CIneg": [precision[1]], "precision_CIpos": [precision[2]], "recall": [recall[0]], "recall_CIneg": [recall[1]], "recall_CIpos": [recall[2]]})

        ## All other rows
        else:
            if train_type!="bin":
                eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "tweedie": tweedie[0], "tweedie_CIneg": tweedie[1], "tweedie_CIpos": tweedie[2], "MSE":mse[0], "MSE_CIneg": mse[1], "MSE_CIpos": mse[2], "F1":f1[0], "F1_CIneg": f1[1], "F1_CIpos": f1[2], "accuracy":accuracy[0], "accuracy_CIneg":accuracy[1], "accuracy_CIpos": accuracy[2], "precision": precision[0], "precision_CIneg": precision[1], "precision_CIpos": precision[2], "recall": recall[0], "recall_CIneg": recall[1], "recall_CIpos": recall[2]}, ignore_index=True)
            else:
                eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "AUC": AUC[0], "AUC_CIneg": AUC[1], "AUC_CIpos": AUC[2], "avPRC":avPRC[0], "avPRC_CIneg": avPRC[1], "avPRC_CIpos": avPRC[2], "F1":f1[0], "F1_CIneg": f1[1], "F1_CIpos": f1[2], "accuracy":accuracy[0], "accuracy_CIneg":accuracy[1], "accuracy_CIpos": accuracy[2], "precision": precision[0], "precision_CIneg": precision[1], "precision_CIpos": precision[2], "recall": recall[0], "recall_CIneg": recall[1], "recall_CIpos": recall[2]}, ignore_index=True)

    ## Only continuous metrics
    else:
        # Results
        if train_type!="bin":
            print("{} {} - Tweedie: {:.2f} ({:.2f}-{:.2f})  MSE: {:.2f} ({:.2f}-{:.2f})".format(set_names[set], group_name, tweedie[0], tweedie[1], tweedie[2], mse[0], mse[1], mse[2]))
            eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name,"GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "tweedie": tweedie[0], "tweedie_CIneg": tweedie[1], "tweedie_CIpos": tweedie[2], "MSE":mse[0], "MSE_CIneg": mse[1], "MSE_CIpos": mse[2]}, ignore_index=True)
        else:
            print("{} {} - AUC: {:.2f} ({:.2f}-{:.2f})  avPRC: {:.2f} ({:.2f}-{:.2f}) ".format(set_names[set], group_name, AUC[0], AUC[1], AUC[2], avPRC[0], avPRC[1], avPRC[2]))
            eval_metrics = eval_metrics._append({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", "N_INDV": n_indv, "N_ABNORM": n_abnorm, "AUC": AUC[0], "AUC_CIneg": AUC[1], "AUC_CIpos": AUC_CIpos[2], "avPRC":avPRC[0], "avPRC_CIneg": avPRC[1], "avPRC_CIpos": avPRC[2]}, ignore_index=True)

    return(eval_metrics)


def eval_subset(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, out_dir=None, out_name=None, subset_name="all", n_boots=500, train_type="cont"):
    eval_metrics = pd.DataFrame()
    all_conf_mats = pd.DataFrame()
    g = plot_observerd_vs_predicted(data, y_cont_goal_col, y_cont_pred_col, y_goal_col, train_type == "bin")
    if out_dir: g.savefig(out_dir + "plots/" + out_name + "_" + get_date() + "_scatter.png", dpi=300)
    g, newcol_x, newcol_y = plot_observed_vs_probability_min5(data, y_cont_goal_col, y_cont_pred_col)
    if out_dir and out_name:
        g.savefig(out_dir + "down/" + get_date() + "/" + out_name + subset_name + "_scatter_min5_" + get_date() + ".png", dpi=300)

    all_conf_mats = conf_matrix_dfs(data, y_goal_col, y_pred_col, all_conf_mats)
    # Metrics with bootstrap for different sets
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=0, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=1, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=2, subset_name=subset_name, n_boots=n_boots, train_type=train_type)

    ## Age-group level
    #eval_metrics, fig = age_group_evals(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, subset_name)
    # if fig_path:
    #     fig.savefig(fig_path + "_" + subset_name + "_agestrata.png", dpi=300)
    #     fig.savefig(fig_path + "_" + subset_name + "_agestrata.pdf")
    
    # ## Time level
    #if "TIME_LAST" in data.columns:
    #    eval_metrics, all_conf_mats, fig = time_evals(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, all_conf_mats, subset_name)
    #     if fig_path:
    #         fig.savefig(fig_path + "_" + subset_name +"_timestrata.png", dpi=300)
    #         fig.savefig(fig_path + "_" + subset_name + "_timestrata.pdf")
    
    return(eval_metrics, all_conf_mats)

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
    pred_data = pred_data.assign(AGE_GROUP=pd.cut(pred_data.EVENT_AGE, [18, 30, 40, 50, 60, 70, 80]))

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
    
#######################################################

def create_report(mdl, out_data, display_scores=[], confusion_labels=["Controls", "Cases"], verbose=True, metric="logloss"):
    """ Reports various metrics of the trained classifier """
    
    dump = dict()
    y_train = out_data.query("SET==0")["TRUE_ABNORM"]
    y_valid = out_data.query("SET==1")["TRUE_ABNORM"]
    print(y_valid.sum())
    train_preds = out_data.query("SET==0")["ABNORM_PREDS"]
    valid_preds = out_data.query("SET==1")["ABNORM_PREDS"]
    print(valid_preds.sum())
    if get_train_type(metric) == "bin":
        y_probs = out_data.query("SET==1")["ABNORM_PROBS"]
        roc_auc = roc_auc_score(y_valid, y_probs)

    train_acc = accuracy_score(y_train, train_preds)
    valid_acc = accuracy_score(y_valid, valid_preds)

    ## Additional scores
    scores_dict = dict()
    for score_name in display_scores:
        func = get_score_func_based_on_metric(score_name)
        scores_dict[score_name] = [func(y_train, train_preds), func(y_valid, valid_preds)]
        
    ## Model Memory
    try:
        model_mem = round(model_memory_size(mdl) / 1024, 2)
    except:
        model_mem = np.nan
    
    logging_print(mdl)
    logging_print("\n=============================> TRAIN-TEST DETAILS <======================================")
    
    ## Metrics
    logging_print(f"Train Size: {len(y_train)} samples")
    logging_print(f"Valid Size: {len(y_valid)} samples")
    logging_print("---------------------------------------------")
    logging_print("Train Accuracy: " + str(train_acc))
    logging_print("Valid Accuracy: " + str(valid_acc))
    logging_print("---------------------------------------------")
    
    if display_scores:
        for k, v in scores_dict.items():
            score_name = ' '.join(map(lambda x: x.title(), k.split('_')))
            logging_print(f'Train {score_name}: ' + str(v[0]))
            logging_print(f'Valid {score_name}: '+ str(v[1]))
            logging_print("")
        logging_print("---------------------------------------------")

    if get_train_type(metric) == "bin":
        logging_print("Area Under ROC (valid): " + str(roc_auc))
        logging_print("---------------------------------------------")
    logging_print(f"Model Memory Size: {model_mem} kB")
    logging_print("\n=============================> CLASSIFICATION REPORT <===================================")
    
    ## Classification Report
    mdl_rep = classification_report(y_valid, valid_preds, output_dict=True)
    
    logging_print(classification_report(y_valid, valid_preds, target_names=confusion_labels))

    ## Dump to report_dict
    dump = dict(mdl=mdl, 
                accuracy=[train_acc, valid_acc], 
                **scores_dict,
                fids=out_data.FINNGENID,
                y_valid=y_valid,
                train_preds=train_preds,
                valid_preds=valid_preds,
                valid_probs=y_probs, 
                report=mdl_rep, 
                roc_auc=roc_auc, 
                model_memory=model_mem)
    return dump


def compare_models(mdl_reports=[], labels=[], score='accuracy', pos_label="1.0"):
    """ Compare evaluation metrics for the True Positive class [1] of 
        binary classifiers passed in the argument and plot ROC and PR curves.
        
        Arguments:
        ---------
         score: is the name corresponding to the sklearn metrics
        
        Returns:
        -------
        compare_table: pandas DataFrame containing evaluated metrics
                  fig: `matplotlib` figure object with ROC and PR curves """

    
    ## Classifier Labels
    default_names = [rep['mdl'].__class__.__name__ for rep in mdl_reports]
    mdl_names = labels if len(labels) == len(mdl_reports) else default_names
    
    ## Compare Table
    table = dict()
    index = ['Train ' + score, 'Valid ' + score, 'Overfitting', 'ROC Area', 'Precision', 'Recall', 'F1-score', 'Support']
    for i in range(len(mdl_reports)):
        scores = [round(i, 3) for i in mdl_reports[i][score]]
        
        roc_auc = mdl_reports[i]['roc_auc']
        
        # Get metrics of True Positive class from sklearn classification_report
        true_positive_metrics = list(mdl_reports[i]['report'][pos_label].values())
        
        table[mdl_names[i]] = scores + [scores[1] < scores[0], roc_auc] + true_positive_metrics
    
    table = pd.DataFrame(data=table, index=index)
    
    
    ## Compare Plots
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)

    # ROC and Precision-Recall
    for i in range(len(mdl_reports)):
        mdl_probs = mdl_reports[i]['valid_probs']
        y_valid = mdl_reports[i]['y_valid']
        roc_plot(y_valid, mdl_probs, label=mdl_names[i], compare=True, ax=ax1)
        plot_calibration(y_valid, mdl_probs, label=mdl_names[i], compare=True, ax1=ax3)
        precision_recall_plot(y_valid, mdl_probs, label=mdl_names[i], compare=True, ax=ax2)
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224)
    #ax4.sharey(ax5)
    plot_box_probs(mdl_reports, mdl_names, ax4, ax5)
    fig.tight_layout()
    plt.close()
    
    return table.T, fig