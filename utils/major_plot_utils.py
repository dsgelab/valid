
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Complex report plots                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import polars as pl
from sklearn.metrics import confusion_matrix
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from code.valid.utils.minor_plot_utils import round_column_min5
def create_report_plots(y_true: Iterable[int], 
                       y_probs: Iterable[float], 
                       y_preds: Iterable[int], 
                       fg_down=False, 
                       train_type="bin", 
                       importance_plot=True, 
                       importances=None, 
                       confusion_labels=["Controls", "Cases"],
                       model_type: str="xgb") -> plt.Figure:
    """Create a report plot with confusion matrix, ROC and Precision-Recall curves, and calibration plot."""
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Figuring out axes # # # # # # # # # # # # # # # # # # # # 
    # Init
    conf_axes = None
    roc_axes = None
    pr_axes = None
    cal_ax1 = None
    cal_ax2 = None
    imp_ax = None
    # Fining correct settings, depending on whether we have importance plot
    if train_type == "bin" or train_type == "multi":
        if importance_plot and importances is not None:
            fig, (row_1, row_2, row_3) = plt.subplots(3, 2, figsize=(11, 10), gridspec_kw={'height_ratios': [1, 1, 0.5]})
            imp_ax, roc_axes = row_1[0], row_1[1]
            cal_ax1, pr_axes = row_2[0], row_2[1]
            cal_ax2, conf_axes = row_3[0], row_3[1]
            cal_ax1.sharex(cal_ax2)
        if imp_ax is None: # remove second row axes
            fig, (row_1, row_2, row_3) = plt.subplots(3, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1, 0.25]})
            conf_axes, roc_axes = row_1[0], row_1[1]
            cal_ax1, pr_axes = row_2[0], row_2[1]
            cal_ax2, delete_axes = row_3[0], row_3[1]
            delete_axes.remove()
            cal_ax1.sharex(cal_ax2)
    if train_type == "cont":
        if importance_plot and importances is not None:
            fig, (row_1, row_2) = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={'height_ratios': [1, 0.3]})
            imp_ax, conf_axes = row_1[0], row_1[1]
            delete_ax, cal_ax = row_2[0], row_2[1]
            delete_ax.remove()
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Plotting # # # # # # # # # # # # # # # # # # # # # # # #
    if imp_ax is not None: 
        if model_type == "xgb" or model_type == "cat":
            feature_importance_plot(importances=importances["mean_shap"], 
                                    feature_labels=importances["labels"], 
                                    ax=imp_ax,
                                    model_type=model_type)
        if model_type == "elr" or model_type == "lr":
            feature_importance_plot(importances=importances["odds_ratio"], 
                                    feature_labels=importances["labels"], 
                                    ax=imp_ax,
                                    model_type=model_type)
    confusion_plot(confusion_matrix(y_true, y_preds), labels=confusion_labels, ax=conf_axes)
    if train_type == "bin" or train_type == "multi":
        ## ROC and Precision-Recall curves
        roc_plot(y_true, y_probs, "Test", ax=roc_axes)
        precision_recall_plot(y_true, y_probs, "Test", ax=pr_axes)
        plot_calibration(y_true, y_probs, cal_ax1, cal_ax2, fg_down=fg_down)
    else:
        if fg_down:
            y_true = np.asarray(y_true)
            y_probs = np.asarray(y_probs)
            safe_cases, _, _ = round_column_min5(y_probs[y_true==1].copy())
            safe_controls, _, _ = round_column_min5(y_probs[y_true==0].copy())
            safe = pl.Series(np.concatenate([safe_cases, safe_controls]))
            labels = pl.concat([pl.Series(y_true[y_true == 1]), pl.Series(y_true[y_true == 0])])
        else:
            safe = pl.Series(y_probs)
            labels = pl.Series(y_true)
        sns.boxplot(x=safe, 
                    y=pd.Categorical(labels.map_elements(lambda x: "Controls" if x == 0 else "Cases", 
                                                         return_dtype=pl.Utf8),
                                     categories=["Controls", "Cases"], ordered=True), 
                    ax=cal_ax, color="black", vert=False, width=.5, whis=(5,95), fill=False, 
                    flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"))
        cal_ax.set_xlabel('Predicted Value')
        cal_ax.set_ylabel('')
    fig.subplots_adjust(wspace=5)
    fig.tight_layout()
    return fig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Single plots                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import shap
import matplotlib.pyplot as plt
def create_shap_dir_plot(shap_values: list[float], 
                         plot_names: list[str]) -> plt.Figure:
    """Create a SHAP summary plot with the given SHAP values and plot names."""
    shap.summary_plot(shap_values, max_display=20, feature_names=plot_names, show=False)
    fig = plt.gcf()
    return fig
    
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from code.valid.utils.minor_plot_utils import pretty_int
def plot_observed_vs_predicted(data: pl.DataFrame, 
                                col_name_x: str, 
                                col_name_y: str,
                                col_name_x_abnorm: str, 
                                prob=True) -> plt.Figure:
    """Plot observed vs predicted values with a regression line and a 1:1 line."""
    sns.set_style('whitegrid')

    # Get the axis limits
    min_x = data.select(pl.col(col_name_x).min()).to_numpy()[0][0]
    max_x = data.select(pl.col(col_name_x).max()).to_numpy()[0][0]
    min_y = data.select(pl.col(col_name_y).min()).to_numpy()[0][0]
    max_y = data.select(pl.col(col_name_y).max()).to_numpy()[0][0]

    axis_limits = [min(min_x, min_y), max(max_x, max_y)]    
    # Show the joint distribution using kernel density estimation
    if prob:
        data=data.with_columns((pl.col(col_name_y)*100).alias(col_name_y))
    g = sns.jointplot(x=data[col_name_x], y=data[col_name_y], kind="scatter", hue=data[col_name_x_abnorm], marginal_kws=dict(common_norm=False))

    g.ax_marg_x.set_xlim(axis_limits[0], axis_limits[1])
    g.ax_joint.set_xlabel("Observed Value")

    if prob: 
        g.ax_joint.set_ylabel("Predicted Probability")
        g.ax_marg_y.set_ylim(0, 100)
        g.ax_joint.text(axis_limits[1]//2, 75, "N = " + pretty_int(data.height) +  "  A = " + pretty_int(data[col_name_x_abnorm].sum()), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
    else: 
        g.ax_joint.set_ylabel("Predicted Value")
        g.ax_marg_y.set_ylim(axis_limits[0], axis_limits[1]) 
        # # This is the x=y line using transforms
        g.ax_joint.plot(axis_limits, axis_limits, "#564D65", linestyle='dashdot', transform=g.ax_joint.transData)
        g.ax_joint.text(axis_limits[1]//4, axis_limits[1]-(0.2*axis_limits[1]), "N = " + pretty_int(data.height) +  "  A = " + pretty_int(data[col_name_x_abnorm].sum()), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))

    return(g)
    
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from code.valid.utils.minor_plot_utils import round_column_min5, pretty_int
def plot_observed_vs_predicted_min5(data: pl.DataFrame, 
                                    col_name_x: str, 
                                    col_name_y: str,
                                    prob=True) -> tuple[plt.Figure, pl.Series, pl.Series]:
    sns.set_style('whitegrid')
    newcol_x, min_val_x, max_val_x = round_column_min5(data[col_name_x])
    if all(data[col_name_y] <= 1.0) and prob: data = data.with_columns(pl.col(col_name_y)*100)
    newcol_y, min_val_y, max_val_y = round_column_min5(data[col_name_y])

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=newcol_x, y=newcol_y, kind="hex", mincnt=5, cmap="twilight_shifted", marginal_kws=dict(stat="density", kde=True, discrete=True, color="#564D65"))
    g.ax_joint.set_xlabel("Observed Value")
    axis_limits = [min(min_val_x, min_val_y)-3, max(max_val_x, max_val_y)+3]
    g.ax_marg_x.set_xlim(axis_limits[0], axis_limits[1])
    if prob:
        g.ax_joint.text(max_val_x-(0.2*max_val_x), 90, "N = " + pretty_int(data.shape[0]), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
        g.ax_marg_y.set_ylim(0, 100)
        g.ax_joint.set_ylabel("Predicted Probability")
    else:
        g.ax_joint.text(3*axis_limits[1]//4, axis_limits[1]-(0.1*axis_limits[1]), "N = " + pretty_int(data.shape[0]), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
        g.ax_joint.set_ylabel("Predicted Value")
        g.ax_joint.plot(axis_limits, axis_limits, "#564D65", linestyle='dashdot', transform=g.ax_joint.transData)

    return(g, newcol_x, newcol_y)

import seaborn as sns
import matplotlib.pyplot as plt
from __future__ import annotations
def confusion_plot(matrix: np.ndarray, 
                   labels=None, 
                   ax=None) -> plt.Axes | plt.Figure:
    """ Display binary confusion matrix as a Seaborn heatmap """
    
    labels = labels if labels else ['Controls', 'Cases']
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    sns.heatmap(data=matrix, cmap='Blues', annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axis, cbar=False, square=True)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('Actual')
    axis.set_title('Confusion Matrix')
    return axis if ax else fig


import matplotlib.pyplot as plt
import seaborn as sns
def confusion_plot(matrix, 
                   labels=None):
    """ Display binary confusion matrix as a Seaborn heatmap """
    
    labels = labels if labels else ['Negative (0)', 'Positive (1)']
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(data=matrix, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    plt.close()
    
    return fig
    
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from __future__ import annotations
from collections.abc import Iterable
def roc_plot(y_true: Iterable[int], 
             y_probs: Iterable[float], 
             label: str, 
             compare=False, 
             ax=None) -> plt.Axes | plt.Figure:
    """ Plot Receiver Operating Characteristic (ROC) curve 
        Set `compare=True` to use this function to compare classifiers. """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    fpr, tpr, thresh = roc_curve(y_true, y_probs, drop_intermediate=False)
    auc = round(roc_auc_score(y_true, y_probs), 2)
    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    label = ' '.join([label, f'({auc})']) if compare else None
    sns.lineplot(x=fpr, y=tpr, ax=axis, label=label)
    
    if compare:
        axis.legend(title='Classifier (AUC)', loc='lower right')
        axis.plot([0,1], [0,1], linestyle='--', color='black')
    else:
        axis.text(0.70, 0.1, f'AUC = { auc }', fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
        # Plot No-Info classifier
        axis.fill_between(fpr, fpr, tpr, alpha=0.3, edgecolor='g', linestyle='--', linewidth=2)
        
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('ROC Curve')
    axis.set_xlabel('False Positive Rate [FPR]\n(1 - Specificity)')
    axis.set_ylabel('True Positive Rate [TPR]\n(Sensitivity or Recall)')
    
    plt.close()
    
    return axis if ax else fig


# Plotting from kaggel https://www.kaggle.com/code/para24/xgboost-stepwise-tuning-using-optuna#3.---Utility-Functions 
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Iterable
from __future__ import annotations
def feature_importance_plot(importances: Iterable[float], 
                            feature_labels: Iterable[str], 
                            ax=None, 
                            n_import=10,
                            model_type: str="xgb") -> plt.Axes | plt.Figure:
    """ Plot feature importances using SHAP values """
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x=importances[0:n_import], y=feature_labels[0:n_import], hue=feature_labels[0:n_import], ax=axis)
    axis.set_title('Feature Importances')
    axis.set_ylabel("")
    if model_type == "xgb" or model_type == "cat":
        axis.set_xlabel("mean(|SHAP value|)")
    elif model_type == "elr" or model_type == "lr":
        axis.set_xlabel("odds ratio")
        axis.set_xlim(left=1)

    plt.close()
    
    return axis if ax else fig
    
import seaborn as sns
import sys
import pandas as pd
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from code.valid.utils.minor_plot_utils import round_column_min5, pretty_int
def plot_box_probs(evals=[], 
                   labels=[], 
                   ax1=None, 
                   ax2=None, 
                   ax_labels=["Controls", "Cases"]) -> None:
    axes = [ax1, ax2]
    for group in [0,1]:
        all_evals = pd.DataFrame()
        for idx, eval_x in enumerate(evals):
            eval_x = pd.DataFrame({"valid_probs": eval_x["valid_probs"].get_column("ABNORM_PROBS"), 
                                   "y_valid": eval_x["y_valid"].get_column("TRUE_ABNORM")})
            eval_crnt = eval_x.loc[eval_x.y_valid==group]
            safe = {"valid_probs": eval_crnt["valid_probs"].copy()*100}
            safe["valid_probs"], min_val, max_val = round_column_min5(safe["valid_probs"])
            safe["group"] = eval_crnt["y_valid"]
            safe["label"] = labels[idx]
            all_evals = pd.concat([all_evals, pd.DataFrame(safe)])
        sns.boxplot(x=all_evals.valid_probs/100, y=all_evals.label, ax=axes[group], vert=False, width=.5, whis=(5,95), hue=all_evals.label, fill=False, flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"))
    try:
        ax1.set_title(ax_labels[0])
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('')
    except:
        pass
    try:
        ax2.set_yticks([])
        ax2.set_title(ax_labels[1])
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('')
    except:
        pass
    
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
def precision_recall_plot(y_true, 
                          y_probs, 
                          label, 
                          compare=False, 
                          ax=None):
    """ Plot Precision-Recall curve.
        Set `compare=True` to use this function to compare classifiers. """
    
    p, r, thresh = precision_recall_curve(y_true, y_probs)
    p, r, thresh = list(p), list(r), list(thresh)
    p.pop()
    r.pop()
    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    
    if compare:
        sns.lineplot(x=r, y=p, ax=axis, label=label)
        axis.set_xlabel('Recall')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')
    else:
        sns.lineplot(x=thresh, y=p, label='Precision', ax=axis)
        axis.set_xlabel('Threshold')
        axis.set_ylabel('Precision')
        axis.legend(loc='lower left')

        axis_twin = axis.twinx()
        sns.lineplot(x=thresh, y=r, color='limegreen', label='Recall', ax=axis_twin)
        axis_twin.set_ylabel('Recall')
        axis_twin.set_ylim(0, 1)
        axis_twin.legend(bbox_to_anchor=(0.24, 0.18))
    
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Precision Vs Recall')
    
    plt.close()
    
    return axis if ax else fig


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Compound plots                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Plotting partially from https://github.com/gerstung-lab/Delphi/blob/main/evaluate_delphi.ipynb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats
from collections.abc import Iterable
import warnings
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from code.valid.utils.minor_plot_utils import round_column_min5
def plot_calibration(y_true: Iterable[int], 
                     y_probs: Iterable[float], 
                     ax1=None, 
                     ax2=None, 
                     bin_type="equal", 
                     compare=False, 
                     label="Calibration", 
                     fg_down=True) -> None:
    """ Plot calibration curve onto axes given as well as a boxplot of the probabilities below on the second axis."""

    if ax1 is None:  fig, (ax1,ax2) = plt.subplots(2,figsize=(15,7), sharex=True, sharey=False, height_ratios=[1, .5])
    if bin_type != "equal": bins = np.quantile(y_probs, np.arange(0,1.1,0.1))
    else: bins=np.arange(0, 1.1, 0.1)

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pred = np.array([y_probs[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])].mean() for b in range(1,len(bins))])
        obs = np.array([y_true[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])].mean()  for b in range(1,len(bins))])
    ci = np.array([scipy.stats.beta(0.1 + y_true[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])].sum(), 0.1 + (1-y_true[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])]).sum()).ppf([0.025,0.975]) for b in range(1,len(bins))])

    # Making sure at least 5 individuals in groups
    sizes = np.array([y_true[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])].shape[0] for b in range(1,len(bins))])
    n_cases = np.array([y_true[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])].sum() for b in range(1,len(bins))])
    n_cntrls = np.array([(y_true[np.logical_and(y_probs > bins[b-1], y_probs <= bins[b])]==0).sum() for b in range(1,len(bins))])
    
    if fg_down:
        valid_bins = np.logical_and(n_cases >= 5, n_cntrls >= 5)
        pred = pred[valid_bins]
        obs = obs[valid_bins]
        ci = ci[valid_bins]
        sizes = sizes[valid_bins]
    if not compare:
        sns.lineplot(x=pred, y=obs, ax=ax1, c="k", label=label)
        ax1.scatter(y_probs.mean(), y_true.mean(), c='r', ec='w', s=80)
    else:
        sns.lineplot(x=pred, y=obs, ax=ax1,label=label)
    
    for j,pr in enumerate(pred):
        if not np.isnan(obs[j]):
            ax1.plot(np.repeat(pr,2),ci[j], lw=1.5, ls=":", c="k")
    sns.scatterplot(x=pred, y=obs, ax=ax1, size=sizes, c="k")
    if compare: ax1.get_legend().remove()

    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, lw=1, c='k', ls="--")
    ax1.set_title('Calibration')
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))

    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Observed Probability')

    if not ax2 is None:
        if fg_down:
            safe_cases, _, _ = round_column_min5(y_probs[y_true==1]*100)
            safe_controls, _, _ = round_column_min5(y_probs[y_true==0]*100)
            safe = pl.Series(np.concatenate([safe_cases, safe_controls]))
            labels = pl.concat([pl.Series(y_true[y_true == 1]), pl.Series(y_true[y_true == 0])])
        else:
            safe = pl.Series((y_probs*100).squeeze())
            labels = pl.Series(y_true.squeeze())
        sns.boxplot(x=safe/100, 
                    y=pd.Categorical(labels.map_elements(lambda x: "Controls" if x == 0 else "Cases", return_dtype=pl.Utf8),categories=["Controls", "Cases"], ordered=True), 
                    ax=ax2, color="black", vert=False, width=.5, whis=(5,95), fill=False, 
                    flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"))
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('')

