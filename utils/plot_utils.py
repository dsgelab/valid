
import seaborn as sns
sns.set_style('whitegrid')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Util functions                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def pretty_int(int_no):
    """Round a number to a pretty string with K (N>1000), M (N>1000000)."""
    if int_no >= 1000000:
        return(str(round(int_no / 1000000,2)) + "M")
    elif int_no >= 1000:
        return(str(round(int_no / 1000)) + "K")
    else:
        return(str(round(int_no, 0)))
    
import polars as pl
from general_utils import read_file
def get_plot_names(col_names: list[str], 
                   lab_name: str,
                   omop_mapping_path="/home/ivm/valid/data/extra_data/upload/kanta_omop_mappings_counts.csv") -> list[str]:
    """Get the plot names for the columns. 
       Prettier/more readable names for output. OMOP mapping is used for lab values."""
    kanta_omop_map = read_file(omop_mapping_path, schema={"measurement_concept_id": pl.Utf8, "concept_name": pl.Utf8})
    kanta_omop_map = dict(zip(kanta_omop_map["measurement_concept_id"].to_list(), kanta_omop_map["concept_name"].to_list()))
    new_names = []
    if lab_name == "hba1c": lab_name = "HbA1c"
    if lab_name == "egfr": lab_name = "eGFR"
    for col_name in col_names:
        if col_name.split("_")[0] in kanta_omop_map.keys():
            if len(col_name.split("_")) > 2:
                new_name = col_name.split("_")[2] + "% " + col_name.split("_")[1].capitalize() + " - " + kanta_omop_map[col_name.split("_")[0]].split("[")[0]
            else:
                new_name = col_name.split("_")[1].capitalize() + " - " + kanta_omop_map[col_name.split("_")[0]].split("[")[0]
        else:
            if col_name.startswith("QUANT"):
                new_name = col_name.split("_")[1] + "% Quant - " + lab_name
            elif col_name in ["MIN", "MEAN", "MAX", "MEDIAN", ]:
                new_name = col_name.capitalize() + " - " + lab_name
            elif col_name == "IDX_QUANT_100":
                new_name = "Last value - " + lab_name
            elif col_name == "IDX_QUANT_50":
                new_name = "Mid value - " + lab_name
            elif col_name == "IDX_QUANT_0":
                new_name = "First value - " + lab_name
            elif col_name == "EVENT_AGE":
                new_name = "Age"
            elif col_name == "FIRST_LAST":
                new_name = "Days between first and last - " + lab_name
            elif col_name == "SEX":
                new_name = "Sex"
            elif col_name == "MIN_LOC":
                new_name = "Location of min (-days) - " + lab_name
            elif col_name == "MAX_LOC":
                new_name = "Location of max (-days) - " + lab_name
            elif col_name == "REG_COEF":
                new_name = "Regression coef - " + lab_name
            elif col_name == "SUM_ABS_CHANGE":
                new_name = "Sum of absolute change - " + lab_name
            elif col_name == "MAX_ABS_CHANGE":
                new_name = "Max of absolute change - " + lab_name
            elif col_name == "MEAN_ABS_CHANGE":
                new_name = "Mean of absolute change - " + lab_name
            elif col_name == "ABNORM":
                new_name = "Number of prior abnorm - " + lab_name
            elif col_name == "MEAN_CHANGE":
                new_name = "Mean change - " + lab_name
            elif col_name == "MAX_CHANGE":
                new_name = "Max change - " + lab_name
            elif col_name == "SUM":
                new_name = "Sum - " + lab_name
            elif col_name == "SD":
                new_name = "Standard deviation - " + lab_name
            elif col_name == "ABS_ENERG":
                new_name = "Absolute energy - " + lab_name
            elif col_name == "SKEW":
                new_name = "Skew - " + lab_name
            elif col_name == "KURT":
                new_name = "Kurtosis - " + lab_name
            elif col_name == "SEQ_LEN":
                new_name = "Number of measurements - " + lab_name
            elif col_name == "YEAR":
                new_name = "Year"
            else:
                new_name = col_name

        new_names.append(new_name)
    return(new_names)
    
import numpy as np
import polars as pl
from collections.abc import Iterable
def round_column_min5(col_data: Iterable) -> tuple[pl.Series, float, float]:
    """Rounds a column to the nearest 5 and replaces values with less than 5 counts with the nearest value with 5 counts.
        Returns the new column and the min and max values of the column."""
    # counting the frequencies of the values
    col_data = pl.Series(col_data)
    mean_freqs = col_data.round().value_counts()
    mean_freqs.columns = ["VALUE", "COUNT"]
    # map for values with too low counts
    value_map = dict()
    # min and max values with at least 5 counts
    min_val = mean_freqs.filter(pl.col("COUNT") >= 5).select(pl.col("VALUE").min()).to_numpy()[0][0]
    max_val = mean_freqs.filter(pl.col("COUNT") >= 5).select(pl.col("VALUE").max()).to_numpy()[0][0]
    # finding mapping
    for row in mean_freqs.rows():
        crnt_value = row[0]
        crnt_count = row[1]
        if crnt_count < 5:
            if abs(min_val-crnt_value) < abs(max_val-crnt_value):
                value_map[crnt_value] = min_val
            else:
                value_map[crnt_value] = max_val
        else: # No new mapping needed
            value_map[crnt_value] = crnt_value
    # mapping the values
    newcol = col_data.map_elements(lambda x: value_map.get(x, x), return_dtype=pl.Float64)

    return(newcol, min_val, max_val)

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
def plot_observerd_vs_predicted(data: pl.DataFrame, 
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
    g = sns.jointplot(x=data[col_name_x], y=data[col_name_y]*100, kind="scatter", hue=data[col_name_x_abnorm])
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
def plot_observerd_vs_predicted_min5(data: pl.DataFrame, 
                                     col_name_x: str, 
                                     col_name_y: str) -> tuple[plt.Figure, pl.Series, pl.Series]:
    """Plot observed vs predicted values but only show values with at least 5 counts.
       Used for downloading plots."""
    sns.set_style('whitegrid')

    newcol_x, min_val_x, max_val_x = round_column_min5(data[col_name_x])
    newcol_y, min_val_y, max_val_y = round_column_min5(data[col_name_y])
    axis_limits = [min(min_val_x, min_val_y)-3, max(max_val_x, max_val_y)+3]
    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=newcol_x, y=newcol_y, kind="hex", mincnt=5, cmap="twilight_shifted", marginal_kws=dict(stat="density", kde=True, discrete=True, color="#564D65"))
    g.ax_marg_x.set_xlim(axis_limits[0], axis_limits[1])
    g.ax_marg_y.set_ylim(axis_limits[0], axis_limits[1])
    g.ax_joint.set_xlabel("observed value")
    g.ax_joint.set_ylabel("predicted value")
    g.ax_joint.text(3*axis_limits[1]//4, axis_limits[1]-(0.1*axis_limits[1]), "N = " + pretty_int(data.shape[0]), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
    
    # # This is the x=y line using transforms
    g.ax_joint.plot(axis_limits, axis_limits, "#564D65", linestyle='dashdot', transform=g.ax_joint.transData)
    return(g, newcol_x, newcol_y)
    
def plot_observed_vs_probability_min5(data: pl.DataFrame, 
                                     col_name_x: str, 
                                     col_name_y: str) -> tuple[plt.Figure, pl.Series, pl.Series]:
    sns.set_style('whitegrid')
    newcol_x, min_val_x, max_val_x = round_column_min5(data[col_name_x])
    if all(data[col_name_y] <= 1.0): data = data.with_columns(pl.col(col_name_y)*100)
    newcol_y, min_val_y, max_val_y = round_column_min5(data[col_name_y])

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=newcol_x, y=newcol_y, kind="hex", mincnt=5, cmap="twilight_shifted", marginal_kws=dict(stat="density", kde=True, discrete=True, color="#564D65"))
    g.ax_joint.text(max_val_x-(0.2*max_val_x), 90, "N = " + pretty_int(data.shape[0]), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
    g.ax_marg_y.set_ylim(0, 100)
    g.ax_joint.set_xlabel("Observed Value")
    g.ax_joint.set_ylabel("Predicted Probability")
    return(g, newcol_x, newcol_y)

import seaborn as sns
from typing import Union
def confusion_plot(matrix: np.ndarray, 
                   labels=None, 
                   ax=None) -> Union[plt.Axes, plt.Figure]:
    """ Display binary confusion matrix as a Seaborn heatmap """
    
    labels = labels if labels else ['Controls', 'Cases']
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    sns.heatmap(data=matrix, cmap='Blues', annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axis, cbar=False, square=True)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('Actual')
    axis.set_title('Confusion Matrix')
    return axis if ax else fig

import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from typing import Union
def roc_plot(y_true: Iterable[int], 
             y_probs: Iterable[float], 
             label: str, 
             compare=False, 
             ax=None) -> Union[plt.Axes, plt.Figure]:
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
from typing import Union
def feature_importance_plot(importances: list[float], 
                            feature_labels: list[str], 
                            ax=None, 
                            n_import=10) -> Union[plt.Axes, plt.Figure]:
    """ Plot feature importances using SHAP values """
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x=importances[0:n_import], y=feature_labels[0:n_import], hue=feature_labels[0:n_import], ax=axis)
    axis.set_title('Feature Importances')
    axis.set_ylabel("")
    axis.set_xlabel("mean(|SHAP value|)")

    plt.close()
    
    return axis if ax else fig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Compound plots                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
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

# Plotting partially from https://github.com/gerstung-lab/Delphi/blob/main/evaluate_delphi.ipynb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections.abc import Iterable
import pandas as pd
import warnings
import pandas as pd
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
        pred = pred[np.logical_and(n_cases>=5, n_cntrls>=5)]
        obs = obs[np.logical_and(n_cases>=5, n_cntrls>=5)]
        ci = ci[np.logical_and(n_cases>=5, n_cntrls>=5)]
        sizes = sizes[np.logical_and(n_cases>=5, n_cntrls>=5)]
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
            safe_cases, _, _ = round_column_min5(y_probs[y_true==1].copy()*100)
            safe_controls, _, _ = round_column_min5( y_probs[y_true==0].copy()*100)
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

# Plotting from kaggel https://www.kaggle.com/code/para24/xgboost-stepwise-tuning-using-optuna#3.---Utility-Functions ######################

def precision_recall_plot(y_true, y_probs, label, compare=False, ax=None):
    """ Plot Precision-Recall curve.
        Set `compare=True` to use this function to compare classifiers. """

    p, r, thresh = precision_recall_curve(y_true, y_probs)
    p, r, thresh = list(p), list(r), list(thresh)
    p.pop()
    r.pop()
    
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)

    aucpr = np.round(skm.average_precision_score(y_true, y_probs),2)
    label = ' '.join([label, f'({aucpr})']) if compare else None

    if compare:
        sns.lineplot(x=r, y=p, ax=axis, label=label)
        axis.set_xlabel('Recall')
        axis.set_ylabel('Precision')
        axis.legend(loc='upper right')
    else:

        sns.lineplot(x=thresh, y=p, label='Precision', ax=axis)
        axis.set_xlabel('Threshold')
        axis.set_ylabel('Precision')
        axis.legend(loc='upper right')

        axis_twin = axis.twinx()
        axis_twin.text(0.70, 0.1, f'avgPrec = { aucpr }', fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
        sns.lineplot(x=thresh, y=r, color='limegreen', label='Recall', ax=axis_twin)
        axis_twin.set_ylabel('Recall')
        axis_twin.set_ylim(0, 1)
        axis_twin.legend(bbox_to_anchor=(0.95, 0.9))
    
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Precision Vs Recall')
    
    plt.close()
    
    return axis if ax else fig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Complex report plots                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from collections.abc import Iterable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def create_report_plots(y_true: Iterable[int], 
                       y_probs: Iterable[float], 
                       y_preds: Iterable[int], 
                       fg_down=False, 
                       train_type="bin", 
                       importance_plot=True, 
                       importances=None, 
                       confusion_labels=["Controls", "Cases"]) -> plt.Figure:
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
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Plotting # # # # # # # # # # # # # # # # # # # # # # # #
    if imp_ax is not None: feature_importance_plot(importances=importances["mean_shap"], 
                                                   feature_labels=importances["labels"], 
                                                   ax=imp_ax)
    confusion_plot(confusion_matrix(y_true, y_preds), labels=confusion_labels, ax=conf_axes)
    if train_type == "bin":
        ## ROC and Precision-Recall curves
        roc_plot(y_true, y_probs, "Test", ax=roc_axes)
        precision_recall_plot(y_true, y_probs, "Test", ax=pr_axes)
        plot_calibration(y_true, y_probs, cal_ax1, cal_ax2, fg_down=fg_down)
        fig.subplots_adjust(wspace=5)
        fig.tight_layout()
        
    return fig

        

###################### Plotting from kaggel https://www.kaggle.com/code/para24/xgboost-stepwise-tuning-using-optuna#3.---Utility-Functions ######################
import timeit
import pickle
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, accuracy_score
from sklearn.exceptions import NotFittedError

import scipy

        
import sklearn.metrics as skm   
