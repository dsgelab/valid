# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Model size                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import sys
import pickle
def model_memory_size(clf):
    """Returns the memory size of a model in KB."""
    try:
        return sys.getsizeof(pickle.dumps(clf))
    except:
        return 0
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Training type and metric                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def get_train_type(metric: str) -> str:
    """Returns the training type based on the metric."""
    if metric == "tweedie": return("cont")
    if metric == "mse": return("cont")
    if metric == "logloss": return("bin")
    if metric == "auc": return("bin")
    if metric == "F1" or metric == "f1" or metric == "aucpr": return("bin")

import sklearn.metrics as skm
def get_score_func_based_on_metric(metric: str) -> callable:
    """Returns the scoring function based on the metric."""
    if metric == "tweedie": return(skm.d2_tweedie_score)
    if metric == "mse": return(skm.mean_squared_error)
    if metric == "logloss": return(skm.log_loss)
    if metric == "auc": return(skm.auc)
    if metric == "F1" or metric == "f1": return(skm.f1_score)
    if metric == "aucpr" or metric == "AUPRC" or metric == "auprc": return(skm.average_precision_score)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Bootstraping                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from collections.abc import Iterable
import numpy as np
def bootstrap_metric(metric_func: callable, 
                     obs: Iterable[float], 
                     preds: Iterable[float], 
                     n_boots=500, 
                     rng_seed=42) -> tuple[np.float64, np.float64, np.float64]:
    """Bootstrapping metrics by shuffling observations and predictions through sampling with redrawing."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Setting up # # # # # # # # # # # # # # # # # # # # # # # 
    bootstraps = []
    obs = np.asarray(obs)
    preds = np.asarray(preds)
    rng = np.random.RandomState(rng_seed)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Point estiamte # # # # # # # # # # # # # # # # # # # # # 
    try: # tweedie sometimes has issues
        total_est = metric_func(obs, preds)
    except: # return nan if not possible
        return(np.nan, np.nan, np.nan)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Bootstrapping # # # # # # # # # # # # # # # # # # # # # # 
    for i in np.arange(n_boots):
        idxs = rng.randint(0, len(obs), len(obs))
        try: # tweedie sometimes has issues
            bootstraps.append(metric_func(obs[idxs], preds[idxs]))
        except:
            continue
    if len(bootstraps) > 0: # if at least one bootstrap was successful
        # Sorting
        bootstraps = np.array(bootstraps)
        bootstraps.sort()
        # 95% CI
        ci_low = bootstraps[int(0.025*len(bootstraps))]
        ci_high = bootstraps[int(0.975*len(bootstraps))]

        return(total_est, ci_low, ci_high)
    else: # if no bootstrap was successful
        return(total_est, np.nan, np.nan)
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Evaluation metrics                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import sklearn.metrics as skm
import polars as pl
def get_optim_precision_recall_cutoff(out_data: pl.DataFrame) -> float:
    """Returns the optimal probability cutoff for the precision-recall curve."""

    precision_, recall_, proba = skm.precision_recall_curve(out_data["TRUE_ABNORM"].to_numpy(), out_data["ABNORM_PROBS"].to_numpy())
    optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]

    return optimal_proba_cutoff

set_names = {0: "Train", 1: "Valid", 2: "Test"}
import polars as pl
def set_metrics(pred_data: pl.DataFrame, 
                y_pred_col: str, 
                y_cont_pred_col: str, 
                y_goal_col: str, 
                y_cont_goal_col: str, 
                eval_metrics: pl.DataFrame, 
                set: int, 
                subset_name="all",
                n_boots=500, 
                train_type="cont"):
    """Calculate metrics on given set of data."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Getting set data  # # # # # # # # # # # # # # # # # # # # 
    set_data = pred_data.filter(pl.col("SET") == set)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Simple stats # # # # # # # # # # # # # # # # # # # # # # 
    n_indv = set_data.height
    n_abnorm = set_data[y_goal_col].sum()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Bootstrapping metrics of interest # # # # # # # # # # # # 
    if train_type != "bin":
        # Measurement for continuous value
        mse = bootstrap_metric(skm.mean_squared_error, set_data[y_cont_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)
        tweedie = bootstrap_metric(lambda x, y: skm.d2_tweedie_score(x, y, power=3), set_data[y_cont_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)
    else:
        AUC = bootstrap_metric(skm.roc_auc_score, set_data[y_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)
        avPRC = bootstrap_metric(skm.average_precision_score, set_data[y_goal_col], set_data[y_cont_pred_col], n_boots=n_boots)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Saving and printing # # # # # # # # # # # # # # # # # # # 
    ## Making sure at least one abnormal is present, otherwise metrics make no sense
    if n_abnorm > 0:
        accuracy = bootstrap_metric(skm.accuracy_score, set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)
        f1 = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)
        precision = bootstrap_metric(lambda x, y: skm.precision_score(x, y, zero_division=0), set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)
        recall = bootstrap_metric(lambda x, y: skm.recall_score(x, y, zero_division=0), set_data[y_goal_col], set_data[y_pred_col], n_boots=n_boots)

        # Results
        if train_type != "bin":
            print("{} - Tweedie: {:.2f} ({:.2f}-{:.2f})  MSE: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})   acurracy: {:.2f} ({:.2f}-{:.2f})   precision {:.2f} ({:.2f}-{:.2f})    recall {:.2f} ({:.2f}-{:.2f})".format(set_names[set], tweedie[0], tweedie[1], tweedie[2], mse[0], mse[1], mse[2], f1[0], f1[1], f1[2], accuracy[0], accuracy[1], accuracy[2], precision[0], precision[1], precision[2], recall[0], recall[1], recall[2]))
        else:
            print("{} - AUC: {:.2f} ({:.2f}-{:.2f})  avPRC: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})   acurracy: {:.2f} ({:.2f}-{:.2f})   precision {:.2f} ({:.2f}-{:.2f})    recall {:.2f} ({:.2f}-{:.2f})".format(set_names[set], AUC[0], AUC[1], AUC[2], avPRC[0], avPRC[1], avPRC[2], f1[0], f1[1], f1[2], accuracy[0], accuracy[1], accuracy[2], precision[0], precision[1], precision[2], recall[0], recall[1], recall[2]))

        ## First row in dataset
        # Create an empty eval_metrics DataFrame
        if train_type != "bin":
            new_row_df = pl.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", 
                                            "N_INDV": n_indv, "N_ABNORM": n_abnorm, 
                                            "tweedie": tweedie[0], "tweedie_CIneg": tweedie[1], "tweedie_CIpos": tweedie[2], 
                                            "MSE":mse[0], "MSE_CIneg": mse[1], "MSE_CIpos": mse[2], 
                                            "F1":f1[0], "F1_CIneg": f1[1], "F1_CIpos": f1[2], 
                                            "accuracy":accuracy[0], "accuracy_CIneg":accuracy[1], "accuracy_CIpos": accuracy[2], 
                                            "precision": precision[0], "precision_CIneg": precision[1], "precision_CIpos": precision[2], 
                                            "recall": recall[0], "recall_CIneg": recall[1], "recall_CIpos": recall[2]})
        else:
            new_row_df = pl.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", 
                                             "N_INDV": n_indv, "N_ABNORM": n_abnorm, 
                                             "AUC": AUC[0], "AUC_CIneg": AUC[1], "AUC_CIpos": AUC[2], 
                                             "avPRC":avPRC[0], "avPRC_CIneg": avPRC[1], "avPRC_CIpos": avPRC[2], 
                                             "F1":f1[0], "F1_CIneg": f1[1], "F1_CIpos": f1[2], 
                                             "accuracy":accuracy[0],"accuracy_CIneg":accuracy[1], "accuracy_CIpos": accuracy[2], 
                                             "precision": precision[0], "precision_CIneg": precision[1], "precision_CIpos": precision[2], 
                                             "recall": recall[0], "recall_CIneg": recall[1], "recall_CIpos": recall[2]})
        if eval_metrics.height == 0:
            eval_metrics = new_row_df
        else:
            eval_metrics = pl.concat([eval_metrics, new_row_df])

    ## Only continuous metrics when there is no abnormal prediciton in the set
    else:
        # Results
        if train_type!="bin":
            print("{} {} - Tweedie: {:.2f} ({:.2f}-{:.2f})  MSE: {:.2f} ({:.2f}-{:.2f})".format(set_names[set], "all", tweedie[0], tweedie[1], tweedie[2], mse[0], mse[1], mse[2]))
            new_row_df = pl.DataFrame({"SET": set_names[set], "SUBSET": subset_name,"GROUP": "all", 
                                       "N_INDV": n_indv, "N_ABNORM": n_abnorm, 
                                       "tweedie": tweedie[0], "tweedie_CIneg": tweedie[1], "tweedie_CIpos": tweedie[2], 
                                       "MSE":mse[0], "MSE_CIneg": mse[1], "MSE_CIpos": mse[2]})
        else:
            print("{} {} - AUC: {:.2f} ({:.2f}-{:.2f})  avPRC: {:.2f} ({:.2f}-{:.2f}) ".format(set_names[set], "all", AUC[0], AUC[1], AUC[2], avPRC[0], avPRC[1], avPRC[2]))
            new_row_df = pl.DataFrame({"SET": set_names[set], "SUBSET": subset_name, "GROUP": "all", 
                                       "N_INDV": n_indv, "N_ABNORM": n_abnorm, 
                                       "AUC": AUC[0], "AUC_CIneg": AUC[1], "AUC_CIpos": AUC[2], 
                                       "avPRC":avPRC[0], "avPRC_CIneg": avPRC[1], "avPRC_CIpos": avPRC[2]})
        if eval_metrics.height == 0:
            eval_metrics = new_row_df
        else:
            eval_metrics = pl.concat([eval_metrics, new_row_df])

    return(eval_metrics)

import polars as pl
def conf_matrix_dfs(pred_data, y_goal_col, y_pred_col, all_conf_mats):
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=0)
    crnt_conf_mats = pl.DataFrame({"all": cm[0].flatten(), "SET": set_names[0]})
    all_conf_mats = pl.concat([all_conf_mats, crnt_conf_mats])
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=1)
    crnt_conf_mats = pl.DataFrame({"all": cm[0].flatten(), "SET": set_names[1]})
    all_conf_mats = pl.concat([all_conf_mats, crnt_conf_mats])
    cm = conf_matrix(pred_data, y_goal_col, y_pred_col, set=2)
    crnt_conf_mats = pl.DataFrame({"all": cm[0].flatten(), "SET": set_names[2]})
    all_conf_mats = pl.concat([all_conf_mats, crnt_conf_mats])
    
    return(all_conf_mats)

import sklearn.metrics as skm
def conf_matrix(pred_data, y_goal_col, y_pred_col, set=1, title=""):
    set_data = pred_data.filter(pl.col("SET") == set)
    set_data = set_data.filter(pl.col(y_goal_col).is_not_null() & pl.col(y_pred_col).is_not_null())

    cm_norm = skm.confusion_matrix(set_data.get_column(y_goal_col).to_numpy(),
                                   set_data.get_column(y_pred_col).to_numpy(), 
                                   normalize="true")
    cm = skm.confusion_matrix(set_data.get_column(y_goal_col).to_numpy(),
                              set_data.get_column(y_pred_col).to_numpy())
    return(cm, cm_norm)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Greater eval functions                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from general_utils import get_date
from plot_utils import plot_observerd_vs_predicted, plot_observed_vs_probability_min5
import polars as pl
def eval_subset(data: pl.DataFrame, 
                y_pred_col: str, 
                y_cont_pred_col: str, 
                y_goal_col: str, 
                y_cont_goal_col: str, 
                plot_path=None, 
                down_path=None, 
                subset_name="all", 
                n_boots=500, 
                train_type="cont") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate the model on the given subset of data.
       Creates plots of observed vs. predicted, observed vs. probability, and confusion matrices.
       Returns the evaluation metrics and confusion matrices."""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Setup # # # # # # # # # # # # # # # # # # # # # # # # # # 
    eval_metrics = pl.DataFrame()
    all_conf_mats = pl.DataFrame()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Plotting # # # # # # # # # # # # # # # # # # # # # # # # 
    if y_cont_goal_col in data.columns:
        g = plot_observerd_vs_predicted(data, y_cont_goal_col, y_cont_pred_col, y_goal_col, train_type == "bin")
        if plot_path: g.savefig(plot_path + subset_name + "_scatter_" + get_date() + ".png", dpi=300)
        g, _, _ = plot_observed_vs_probability_min5(data, y_cont_goal_col, y_cont_pred_col)
        if down_path:
            g.savefig(down_path + subset_name + "_scatter_min5_" + get_date() + ".png", dpi=300)

    all_conf_mats = conf_matrix_dfs(data, y_goal_col, y_pred_col, all_conf_mats)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Bootstrapped metrics# # # # # # # # # # # # # # # # # # #     
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=0, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=1, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    eval_metrics = set_metrics(data, y_pred_col, y_cont_pred_col, y_goal_col, y_cont_goal_col, eval_metrics, set=2, subset_name=subset_name, n_boots=n_boots, train_type=train_type)
    
    return(eval_metrics, all_conf_mats)

from general_utils import logging_print
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
def create_report(mdl: object, 
                  out_data: pl.DataFrame, 
                  display_scores=[], 
                  confusion_labels=["Controls", "Cases"], 
                  metric="logloss"):
    """ Reports various metrics of the trained classifier.
        Returns a dictionary containing the model, accuracy, scores, predictions, and classification report.
        Output:
        -------
        dump: dictionary containing the model, accuracy, scores, predictions, and classification report.
                    - mdl: trained model
                    - accuracy: list of train and valid accuracy
                    - scores: dictionary of train and valid scores
                    - fids: list of FINNGENIDs
                    - y_valid: list of true labels
                    - train_preds: list of train predictions
                    - valid_preds: list of valid predictions
                    - valid_probs: list of valid probabilities
                    - report: classification report
                    - roc_auc: area under ROC curve
                    - model_memory: model memory size in KB
        """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Get data # # # # # # # # # # # # # # # # # # # # # # # #    
    y_train = out_data.filter(pl.col("SET")==0).select("TRUE_ABNORM")
    y_valid = out_data.filter(pl.col("SET")==1).select("TRUE_ABNORM")
    train_preds = out_data.filter(pl.col("SET")==0).select("ABNORM_PREDS")
    valid_preds = out_data.filter(pl.col("SET")==1).select("ABNORM_PREDS")
    if get_train_type(metric) == "bin":
        y_probs = out_data.filter(pl.col("SET")==1).select("ABNORM_PROBS")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Metrics # # # # # # # # # # # # # # # # # # # # # # # # #  
        roc_auc = roc_auc_score(y_valid, y_probs) # defined only for binary classification
    train_acc = accuracy_score(y_train, train_preds)
    valid_acc = accuracy_score(y_valid, valid_preds)
    # Additional scores
    scores_dict = dict()
    for score_name in display_scores:
        func = get_score_func_based_on_metric(score_name)
        scores_dict[score_name] = [func(y_train, train_preds), func(y_valid, valid_preds)]   
    # Model Memory
    try:
        model_mem = round(model_memory_size(mdl) / 1024, 2)
    except:
        model_mem = np.nan
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Logging # # # # # # # # # # # # # # # # # # # # # # # # #  
    logging_print(mdl)
    logging_print("\n=============================> TRAIN-TEST DETAILS <======================================")
    # Metrics
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
                fids=out_data["FINNGENID"],
                y_valid=y_valid,
                train_preds=train_preds,
                valid_preds=valid_preds,
                valid_probs=y_probs, 
                report=mdl_rep, 
                roc_auc=roc_auc, 
                model_memory=model_mem)
    return dump

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from plot_utils import roc_plot, precision_recall_plot, plot_calibration, plot_box_probs
def compare_models(mdl_reports=[], 
                   labels=[], 
                   score='accuracy', 
                   pos_label="1.0") -> tuple[pl.DataFrame, plt.Figure]:
    """ Compare evaluation metrics for the True Positive class [1] of 
        binary classifiers passed in the argument and plot ROC and PR curves.
        
        Arguments:
        ---------
         score: is the name corresponding to the sklearn metrics
        
        Returns:
        -------
        compare_table: pandas DataFrame containing evaluated metrics
                  fig: `matplotlib` figure object with ROC and PR curves """

    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Table # # # # # # # # # # # # # # # # # # # # # # # # # #  
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
    
    table = (pl.DataFrame(table)
             .with_columns(pl.Series("index", index))
             .unpivot(index="index", variable_name="Model")
             .pivot("index", index="Model")
    )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Plotting # # # # # # # # # # # # # # # # # # # # # # # #    
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
    
    return table, fig