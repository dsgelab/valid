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
    if metric in ["q10", "q05", "q25", "q50", "q75", "q90", "q95"]: return("cont")
    if metric == "mae": return("cont")
    if metric == "mse" or metric == "rmse": return("cont")
    if metric == "logloss": return("bin")
    if metric == "mlogloss": return("multi")
    if metric == "auc": return("bin")
    if metric == "F1" or metric == "f1" or metric == "aucpr": return("bin")
    else:
        raise(ValueError("Unknown metric: {}".format(metric)))

import sklearn.metrics as skm
def get_score_func_based_on_metric(metric: str) -> callable:
    """Returns the scoring function based on the metric."""
    if metric == "q10": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.1))
    if metric == "q05": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.05))
    if metric == "q25": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.25))
    if metric == "q50": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.50))
    if metric == "q75": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.75))
    if metric == "q90": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.90))
    if metric == "q95": return(lambda x, y: skm.mean_pinball_loss(y,x, alpha=0.95))
    if metric == "tweedie": return(skm.d2_tweedie_score)
    if metric == "mse": return(skm.mean_squared_error)
    if metric == "mae": return(skm.mean_absolute_error)
    if metric == "rmse": return(skm.root_mean_squared_error)
    if metric == "logloss": return(skm.log_loss)
    if metric == "auc": return(skm.auc)
    if metric == "F1" or metric == "f1": return(skm.f1_score)
    if metric == "aucpr" or metric == "AUPRC" or metric == "auprc": return(skm.average_precision_score)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Custom functions                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def continuous_nri(y_true, old_probs, new_probs):
    y_true = np.array(y_true)
    old_probs = np.array(old_probs)
    new_probs = np.array(new_probs)
    
    # Determine reclassification
    delta = new_probs - old_probs

    # Events (y=1)
    events = y_true == 1
    up_event = np.sum(delta[events] > 0) / np.sum(events)
    down_event = np.sum(delta[events] < 0) / np.sum(events)

    # Non-events (y=0)
    nonevents = y_true == 0
    up_nonevent = np.sum(delta[nonevents] > 0) / np.sum(nonevents)
    down_nonevent = np.sum(delta[nonevents] < 0) / np.sum(nonevents)

    nri = (up_event - down_event) + (down_nonevent - up_nonevent)
    return nri
    
def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        # Bin mask
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i+1])
        if np.any(mask):
            bin_prob = y_prob[mask]
            bin_true = y_true[mask]
            bin_conf = bin_prob.mean()
            bin_acc = bin_true.mean()
            ece += (len(bin_true) / n) * abs(bin_acc - bin_conf)
    return ece

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

from collections.abc import Iterable
import numpy as np
def bootstrap_nri(metric_func: callable, 
                  obs: Iterable[float], 
                  preds_old: Iterable[float], 
                  preds_new: Iterable[float], 
                  n_boots=500, 
                  rng_seed=42) -> tuple[np.float64, np.float64, np.float64]:
    """Bootstrapping metrics by shuffling observations and predictions through sampling with redrawing."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Setting up # # # # # # # # # # # # # # # # # # # # # # # 
    bootstraps = []
    obs = np.asarray(obs)
    preds_old = np.asarray(preds_old)
    preds_new = np.asarray(preds_new)
    rng = np.random.RandomState(rng_seed)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Point estiamte # # # # # # # # # # # # # # # # # # # # # 
    try: # tweedie sometimes has issues
        total_est = metric_func(obs, preds_old, preds_new)
    except: # return nan if not possible
        return(np.nan, np.nan, np.nan)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Bootstrapping # # # # # # # # # # # # # # # # # # # # # # 
    for i in np.arange(n_boots):
        idxs = rng.randint(0, len(obs), len(obs))
        try: # tweedie sometimes has issues
            bootstraps.append(metric_func(obs[idxs], preds_old[idxs], preds_new[idxs]))
        except:
            continue
    if len(bootstraps) > 0: # if at least one bootstrap was successful
        # Sorting
        bootstraps = np.array(bootstraps)
        bootstraps.sort()
        # 95% CI
        ci_low = bootstraps[int(0.025*len(bootstraps))]
        ci_high = bootstraps[int(0.975*len(bootstraps))]
        p_value = np.mean(np.abs(bootstraps)>=np.abs(total_est))

        return(total_est, ci_low, ci_high, p_value)
    else: # if no bootstrap was successful
        return(total_est, np.nan, np.nan, np.nan)

from collections.abc import Iterable
import numpy as np
def bootstrap_difference(metric_func: callable, 
                         preds_1: Iterable[float], 
                         preds_2: Iterable[float], 
                         obs: Iterable[float],
                         n_boots=500, 
                         rng_seed=42) -> tuple[np.float64, np.float64, np.float64]:
    """Bootstrapping metrics by shuffling observations and predictions through sampling with redrawing."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Setting up # # # # # # # # # # # # # # # # # # # # # # # 
    bootstraps_diffs = []
    preds_1 = np.asarray(preds_1)
    preds_2 = np.asarray(preds_2)
    obs = np.asarray(obs)
    rng = np.random.RandomState(rng_seed)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Point estiamte # # # # # # # # # # # # # # # # # # # # # 
    try: # tweedie sometimes has issues
        total_est_1 = metric_func(obs, preds_1)
        total_est_2 = metric_func(obs, preds_2)
        total_diff = total_est_2 - total_est_1
    except: # return nan if not possible
        return(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Bootstrapping # # # # # # # # # # # # # # # # # # # # # # 
    for i in np.arange(n_boots):
        idxs = rng.randint(0, len(obs), len(obs))
        try: # tweedie sometimes has issues
            est_1 = metric_func(obs[idxs], preds_1[idxs])
            est_2 = metric_func(obs[idxs], preds_2[idxs])
            bootstraps_diffs.append(est_2-est_1)
        except:
            continue
    if len(bootstraps_diffs) > 0: # if at least one bootstrap was successful
        # Sorting
        bootstraps_diffs = np.array(bootstraps_diffs)
        bootstraps_diffs.sort()
        # 95% CI
        ci_low = bootstraps_diffs[int(0.025*len(bootstraps_diffs))]
        ci_high = bootstraps_diffs[int(0.975*len(bootstraps_diffs))]
        # Probability of observing as strong difference across bootstraps
        null_diffs = total_diff-bootstraps_diffs
        p_value = np.mean(np.abs(null_diffs)>=np.abs(bootstraps_diffs))
        return(total_diff, ci_low, ci_high, p_value, total_est_1, total_est_2)
    else: # if no bootstrap was successful
        return(total_est, np.nan, np.nan, np.nan, np.nan, np.nan)
    
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
def get_all_eval_metrics(data: pl.DataFrame,
                         plot_path: str,
                         down_path: str,
                         n_boots: int,
                         train_type=str) -> pl.DataFrame:
    if "y_MEAN_ABNORM" in data.columns and "y_DIAG" in data.columns and "y_MIN_ABNORM" in data.columns and "y_NEXT_ABNORM" in data.columns:
        print("Diag")
        eval_metrics = eval_subset(data=data, 
                                   y_pred_col="ABNORM_PREDS", 
                                   y_cont_pred_col="ABNORM_PROBS", 
                                   y_goal_col="y_DIAG", 
                                   y_cont_goal_col="y_MEAN", 
                                   plot_path=plot_path, 
                                   down_path=down_path, 
                                   subset_name="all", 
                                   n_boots=n_boots, 
                                   train_type=train_type)
        eval_metrics = eval_metrics.insert_column(3, pl.Series("GOAL", ["y_DIAG"]*eval_metrics.height))

        print("\nMean abnorm")
        crnt_eval_metrics = eval_subset(data=data, 
                                        y_pred_col="ABNORM_PREDS", 
                                        y_cont_pred_col="ABNORM_PROBS", 
                                        y_goal_col="y_MEAN_ABNORM", 
                                        y_cont_goal_col="y_MEAN", 
                                        plot_path=plot_path, 
                                        down_path=down_path, 
                                        subset_name="all", 
                                        n_boots=n_boots, 
                                        train_type=train_type)
        crnt_eval_metrics = crnt_eval_metrics.insert_column(3, pl.Series("GOAL", ["y_MEAN_ABNORM"]*crnt_eval_metrics.height))
        eval_metrics = pl.concat([eval_metrics, crnt_eval_metrics])

        print("\nNext abnorm")
        crnt_eval_metrics = eval_subset(data=data, 
                                        y_pred_col="ABNORM_PREDS", 
                                        y_cont_pred_col="ABNORM_PROBS", 
                                        y_goal_col="y_NEXT_ABNORM", 
                                        y_cont_goal_col="y_NEXT", 
                                        plot_path=plot_path, 
                                        down_path=down_path, 
                                        subset_name="all", 
                                        n_boots=n_boots, 
                                        train_type=train_type)
        crnt_eval_metrics = crnt_eval_metrics.insert_column(3, pl.Series("GOAL", ["y_NEXT_ABNORM"]*crnt_eval_metrics.height))
        eval_metrics = pl.concat([eval_metrics, crnt_eval_metrics])
    else:
        eval_metrics = eval_subset(data=data, 
                                   y_pred_col="ABNORM_PREDS", 
                                   y_cont_pred_col="ABNORM_PROBS", 
                                   y_goal_col="TRUE_ABNORM", 
                                   y_cont_goal_col="TRUE_VALUE", 
                                   plot_path=plot_path, 
                                   down_path=down_path, 
                                   subset_name="all", 
                                   n_boots=n_boots, 
                                   train_type=train_type)

    return(eval_metrics)

from plot_utils import create_report_plots, plot_observed_vs_predicted, plot_observed_vs_predicted_min5
from general_utils import get_date, make_dir

def save_all_report_plots(out_data: pl.DataFrame,
                          out_plot_path: str,
                          out_down_path: str,
                          train_importances: pl.DataFrame=None,
                          valid_importances: pl.DataFrame=None,
                          test_importances: pl.DataFrame=None,
                          train_type: str ="bin") -> None:
    set_names = {0: "train", 1: "valid", 2: "test"}
    for goal in ["y_MEAN_ABNORM", "y_NEXT_ABNORM", "y_MIN_ABNORM", "y_DIAG"]:
        if goal in out_data.columns:
            for set_no in [0,1,2]:
                if set_no == 0: crnt_importances = train_importances
                elif set_no == 1: crnt_importances = valid_importances
                elif set_no == 2: crnt_importances = test_importances
        
                if out_data[goal].unique().len()==2 or train_type == "cont":
                    if train_type == "cont" and out_data[goal].unique().len()>2:
                        out_data = out_data.with_columns(pl.when(pl.col(goal)==1).then(pl.lit(1)).otherwise(pl.lit(0)).alias(goal))
                        
                    ##### 6 panel report plots""
                    fig = create_report_plots(out_data.filter(pl.col.SET == set_no).select(goal), 
                                              out_data.filter(pl.col.SET == set_no).select("ABNORM_PROBS"),
                                              out_data.filter(pl.col.SET == set_no).select("ABNORM_PREDS"),
                                              importances=crnt_importances,
                                              train_type=train_type,
                                              fg_down=False)
                    make_dir(out_plot_path+"/"+goal+"/")
                    fig.savefig(out_plot_path+"/"+goal+"/"+set_names[set_no]+"_report_" + get_date() + ".png")
                    fig = create_report_plots(out_data.filter(pl.col.SET == set_no).select(goal), 
                                              out_data.filter(pl.col.SET == set_no).select("ABNORM_PROBS"),
                                              out_data.filter(pl.col.SET == set_no).select("ABNORM_PREDS"),
                                              importances=crnt_importances,
                                              train_type=train_type,
                                              fg_down=True)
                    fig.savefig(out_down_path+goal+"_"+set_names[set_no]+"_report_" + get_date() + ".png")
                    fig.savefig(out_down_path+goal+"_"+set_names[set_no]+"_report_" + get_date() + ".pdf")
                    ##### prediction distibution plots""
                    fig = plot_observed_vs_predicted(out_data.filter(pl.col.SET == set_no),
                                                   col_name_x=goal.replace("_ABNORM", ""), 
                                                   col_name_y="ABNORM_PROBS", 
                                                   col_name_x_abnorm=goal, 
                                                   prob = (train_type=="bin"))
                    
                    fig.savefig(out_plot_path+"/"+goal+"/"+set_names[set_no] + "_scatter_" + get_date() + ".png", dpi=300)
                    plt.close()
                    fig, _, _ = plot_observed_vs_predicted_min5(out_data.filter(pl.col.SET == set_no),
                                                                col_name_x=goal.replace("_ABNORM", ""), 
                                                                col_name_y="ABNORM_PROBS", 
                                                                prob = (train_type=="bin"))
                    fig.savefig(out_down_path+goal+"_"+set_names[set_no] + "_scatter_" + get_date() + ".png", dpi=300)
                    plt.close()
                else:
                    crnt_importances = pl.DataFrame(crnt_importances).with_columns(pl.col(crnt_importances.columns[0]).arr.to_struct().alias(crnt_importances.columns[0])).unnest(crnt_importances.columns[0])
        
                    for abnorm_type in out_data[goal].unique():
                        if abnorm_type == 0:
                            two_abnorm = (pl.when(pl.col(goal)==0).then(pl.lit(1)).otherwise(pl.lit(0)).alias(goal))
                        else:
                            two_abnorm=((pl.when((pl.col(goal)!=1)&(pl.col(goal)!=0)).then(pl.lit(0)).otherwise(pl.col(goal)).alias(goal)))

                        if "ABNORM_PROBS_"+str(abnorm_type) in out_data.columns:
                            if abnorm_type == 0: crnt_n_importances = crnt_importances.select("field_0", "labels").rename({"field_0": "mean_shap"})
                            elif abnorm_type == 1: crnt_n_importances = crnt_importances.select("field_1", "labels").rename({"field_1": "mean_shap"})
                            elif abnorm_type == -1: crnt_n_importances = crnt_importances.select("field_2", "labels").rename({"field_2": "mean_shap"})
                            crnt_n_importances = crnt_n_importances.sort("mean_shap", descending=True)
                            fig = create_report_plots(out_data.filter(pl.col.SET == set_no).with_columns(two_abnorm).select(goal), 
                                                      out_data.filter(pl.col.SET == set_no).select("ABNORM_PROBS_"+str(abnorm_type)),
                                                      out_data.filter(pl.col.SET == set_no).select("ABNORM_PREDS_"+str(abnorm_type)),
                                                      importances=crnt_n_importances,
                                                      train_type=train_type,
                                                      fg_down=False)
                            make_dir(out_plot_path+"/"+goal+"/abnorm"+str(abnorm_type).replace("-","n")+"/")
                            fig.savefig(out_plot_path+"/"+goal+"/abnorm"+str(abnorm_type).replace("-","n")+"/"+set_names[set_no]+"_report_" + get_date() + ".png")
                            fig = create_report_plots(out_data.filter(pl.col.SET == set_no).with_columns(two_abnorm).select(goal), 
                                                      out_data.filter(pl.col.SET == set_no).select("ABNORM_PROBS_"+str(abnorm_type)),
                                                      out_data.filter(pl.col.SET == set_no).select("ABNORM_PREDS_"+str(abnorm_type)),
                                                      importances=crnt_n_importances,
                                                      train_type=train_type,
                                                      fg_down=True)
                            fig.savefig(out_down_path + "_"+goal+"_abnorm"+str(abnorm_type).replace("-","n")+"_"+set_names[set_no]+"_report_" + get_date() + ".png")
                            ##### prediction distibution plots""
                            fig = plot_observed_vs_predicted(out_data.filter(pl.col.SET == set_no),
                                                        col_name_x=goal.replace("_ABNORM", ""), 
                                                        col_name_y="ABNORM_PROBS", 
                                                        col_name_x_abnorm=goal, 
                                                        prob = (train_type=="bin"))
                            
                            fig.savefig(out_plot_path+"/"+goal+"/"+set_names[set_no] + "_scatter_" + get_date() + ".png", dpi=300)
                            plt.close()
                            fig, _, _ = plot_observed_vs_predicted_min5(out_data.filter(pl.col.SET == set_no),
                                                                        col_name_x=goal.replace("_ABNORM", ""), 
                                                                        col_name_y="ABNORM_PROBS", 
                                                                        prob = (train_type=="bin"))
                            fig.savefig(out_down_path+goal+"_"+set_names[set_no] + "_scatter_" + get_date() + ".png", dpi=300)
                            plt.close()
        
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

import matplotlib.pyplot as plt
import polars as pl
from model_eval_utils import bootstrap_metric
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, recall_score, precision_score, f1_score
from collections import defaultdict
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from plot_utils import roc_plot, precision_recall_plot, plot_calibration, plot_box_probs
def compare_models(probs=dict(),
                   labels=dict(), 
                   score='accuracy', 
                   pos_label="1.0",
                   label_type="bin") -> tuple[pl.DataFrame, plt.Figure]:
    """ Compare evaluation metrics for the True Positive class [1] of 
        binary classifiers passed in the argument and plot ROC and PR curves.
        
        Arguments:
        ---------
         score: is the name corresponding to the sklearn metrics
        
        Returns:
        -------
        compare_table: pandas DataFrame containing evaluated metrics
                  fig: `matplotlib` figure object with ROC and PR curves """

    

    if label_type == "bin":
        ## Compare Table
        score_names = ['ROC Area', "Average Precision", "cut-off",  "Positives", "TP at optimal PR", "F1 (macro)", "F1 (binary)", "Precision", "Recall","FP at optimal PR", "FP at 50% sens"]
        table = defaultdict(lambda: [])
        for score_idx, score_name in enumerate(score_names):
            for model_name, crnt_probs in probs.items():
                if score_name in ['ROC Area', "Average Precision", "F1 (macro)", "F1 (binary)", "Precision", "Recall"]:
                    if score_name == "ROC Area":
                        score, ci_neg, ci_pos = bootstrap_metric(roc_auc_score, labels[model_name], crnt_probs)
                    elif score_name == "Average Precision":
                        score, ci_neg, ci_pos = bootstrap_metric(skm.average_precision_score, labels[model_name], crnt_probs)
                    elif score_name == "F1 (macro)":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                        score, ci_neg, ci_pos = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), labels[model_name], crnt_probs.to_numpy()>optimal_proba_cutoff)
                    elif score_name == "Precision":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                        score, ci_neg, ci_pos = bootstrap_metric(lambda x,y: skm.precision_score(x, y, average="binary", zero_division=0), labels[model_name], (crnt_probs.to_numpy()>optimal_proba_cutoff))
                    elif score_name == "Recall":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                        score, ci_neg, ci_pos = bootstrap_metric(recall_score, labels[model_name], (crnt_probs.to_numpy()>optimal_proba_cutoff))
                    elif score_name == "F1 (binary)":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                        score, ci_neg, ci_pos = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="binary", zero_division=0), labels[model_name], crnt_probs.to_numpy()>optimal_proba_cutoff)

                    else:
                        raise("probelms with score " + score_name)
                    table[model_name].append(str(round(score, 2))+" ("+str(round(ci_neg, 2))+"-"+str(round(ci_pos, 2))+")")
                else:
                    if score_name == "cut-off":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        score = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                    elif score_name == "TP at optimal PR":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                        score = ((crnt_probs.to_numpy()>optimal_proba_cutoff)& labels[model_name]).sum()
                    elif score_name == "FP at optimal PR":
                        precision_, recall_, proba = precision_recall_curve(labels[model_name], crnt_probs)
                        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
                        score = np.logical_and((crnt_probs>optimal_proba_cutoff).to_numpy(),( labels[model_name]==0).to_numpy()).sum()
                    elif score_name == "FP at 50% sens":
                        fpr, tpr, thresholds = roc_curve(labels[model_name], crnt_probs)
                        best = min(tpr, key=lambda x:abs(x-0.5))
                        optimal_proba_cutoff=(max(thresholds[tpr==best]))
                        score = np.logical_and((crnt_probs>optimal_proba_cutoff).to_numpy(),( labels[model_name]==0).to_numpy()).sum()
                    elif score_name == "Positives":
                        score = labels[model_name].sum()
                    table[model_name].append(str(round(score, 2)))
        table = (pl.DataFrame(table)
                  .with_columns(pl.Series("index", score_names))
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
        for model_name, mdl_probs in probs.items():
            roc_plot(labels[model_name], mdl_probs, label=model_name, compare=True, ax=ax1)
            plot_calibration(labels[model_name], mdl_probs, label=model_name, compare=True, ax1=ax3, fg_down=True)
            precision_recall_plot(labels[model_name], mdl_probs, label=model_name, compare=True, ax=ax2)
    
        fig.tight_layout()
        plt.close()
    else:
        ## Compare Table
        score_names = ["Positives", "TP", "F1 (macro)", "F1 (binary)", "Precision", "Recall"]
        table = defaultdict(lambda: [])
        for score_idx, score_name in enumerate(score_names):
            for model_name, crnt_probs in probs.items():
                if score_name in ["F1 (macro)", "F1 (binary)", "Precision", "Recall"]:
                    if score_name == "F1 (macro)":
                        score, ci_neg, ci_pos = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), labels[model_name].to_numpy(), probs[model_name].to_numpy())
                    elif score_name == "F1 (binary)":
                        score, ci_neg, ci_pos = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="binary", zero_division=0), labels[model_name].to_numpy(), probs[model_name].to_numpy())
                    elif score_name == "Precision":
                        score, ci_neg, ci_pos = bootstrap_metric(precision_score, labels[model_name], probs[model_name])
                    elif score_name == "Recall":
                        score, ci_neg, ci_pos = bootstrap_metric(recall_score, labels[model_name], probs[model_name])
                    else:
                        raise("probelms with score " + score_name)
                    table[model_name].append(str(round(score, 2))+" ("+str(round(ci_neg, 2))+"-"+str(round(ci_pos, 2))+")")
                else:
                    if score_name == "Positives":
                        score = labels[model_name].sum()
                    elif score_name == "TP":
                        score = np.logical_and(probs[model_name].to_numpy(), labels[model_name].to_numpy()).sum()
                    table[model_name].append(str(score))
        table = (pl.DataFrame(table)
                  .with_columns(pl.Series("index", score_names))
                  .unpivot(index="index", variable_name="Model")
                  .pivot("index", index="Model")
        )
        fig = None
    
    return table, fig