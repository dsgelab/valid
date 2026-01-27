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
        #p_value = np.mean(np.abs(bootstraps)>=np.abs(total_est))

        return(total_est, ci_low, ci_high)
    else: # if no bootstrap was successful
        return(total_est, np.nan, np.nan)
    
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Greater eval functions                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from minor_plot_utils import create_report_plots
from general_utils import get_date, make_dir
def save_all_report_plots(out_data: pl.DataFrame,
                          out_plot_path: str,
                          out_down_path: str,
                          study_name: str,
                          train_importances: pl.DataFrame=None,
                          valid_importances: pl.DataFrame=None,
                          test_importances: pl.DataFrame=None,
                          train_type: str ="bin",
                          model_type: str="xgb",
                          n_features: int=None) -> None:
    set_names = {0: "train", 1: "valid", 2: "test"}
    eval_sets = [0,1,2]
    for goal in ["y_MEAN_ABNORM", "y_NEXT_ABNORM"]:
        if goal in out_data.columns:
            crnt_out_down_path = out_down_path+"/"+goal+"/"; make_dir(crnt_out_down_path)
            for set_no in eval_sets:
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
                                              fg_down=False,
                                              model_type=model_type)
                    make_dir(out_plot_path+"/"+goal+"/")
                    if n_features:
                        fig.savefig(out_plot_path+"/"+goal+"/"+set_names[set_no]+"_report_fs"+str(n_features)+"_" + get_date() + ".png")
                    else:
                        fig.savefig(out_plot_path+"/"+goal+"/"+set_names[set_no]+"_report_" + get_date() + ".png")
                    fig = create_report_plots(out_data.filter(pl.col.SET == set_no).select(goal), 
                                              out_data.filter(pl.col.SET == set_no).select("ABNORM_PROBS"),
                                              out_data.filter(pl.col.SET == set_no).select("ABNORM_PREDS"),
                                              importances=crnt_importances,
                                              train_type=train_type,
                                              fg_down=True,
                                              model_type=model_type)
                    if n_features:
                        fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+set_names[set_no]+"_report_fs"+str(n_features)+"_" + get_date() + ".png")
                        fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+set_names[set_no]+"_report_fs"+str(n_features)+"_" + get_date() + ".pdf")
                    else:
                        fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+set_names[set_no]+"_report_" + get_date() + ".png")
                        fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+set_names[set_no]+"_report_" + get_date() + ".pdf")
                else:
                    crnt_importances = (pl.DataFrame(crnt_importances)
                                        .with_columns(pl.col(crnt_importances.columns[0]).arr.to_struct().alias(crnt_importances.columns[0]))
                                        .unnest(crnt_importances.columns[0])
                                       )
        
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
                                                      fg_down=False,
                                                      model_type=model_type)
                            make_dir(out_plot_path+"/"+goal+"/abnorm"+str(abnorm_type).replace("-","n")+"/")
                            if n_features:
                                fig.savefig(out_plot_path+"/"+goal+"/abnorm"+str(abnorm_type).replace("-","n")+"/"+set_names[set_no]+"_report_fs"+str(n_features)+"_" + get_date() + ".png")
                            else:
                                fig.savefig(out_plot_path+"/"+goal+"/abnorm"+str(abnorm_type).replace("-","n")+"/"+set_names[set_no]+"_report_" + get_date() + ".png")
                            fig = create_report_plots(out_data.filter(pl.col.SET == set_no).with_columns(two_abnorm).select(goal), 
                                                      out_data.filter(pl.col.SET == set_no).select("ABNORM_PROBS_"+str(abnorm_type)),
                                                      out_data.filter(pl.col.SET == set_no).select("ABNORM_PREDS_"+str(abnorm_type)),
                                                      importances=crnt_n_importances,
                                                      train_type=train_type,
                                                      fg_down=True,
                                                      model_type=model_type)
                            if n_features:
                                fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+"abnorm"+str(abnorm_type).replace("-","n")+"_"+set_names[set_no]+"_report_fs"+str(n_features)+"_" + get_date() + ".png")
                                fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+"abnorm"+str(abnorm_type).replace("-","n")+"_"+set_names[set_no]+"_report_fs"+str(n_features)+"_" + get_date() + ".pdf")
                            else:
                                fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+"abnorm"+str(abnorm_type).replace("-","n")+"_"+set_names[set_no]+"_report_" + get_date() + ".png")
                                fig.savefig(crnt_out_down_path+goal+"_"+study_name+"_"+"abnorm"+str(abnorm_type).replace("-","n")+"_"+set_names[set_no]+"_report_" + get_date() + ".pdf")
        
