
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import sklearn.metrics as skm   
import scipy

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
                   lab_name_two: str="",
                   omop_mapping_path="/home/ivm/valid/data/extra_data/upload/kanta_omop_mappings_counts.csv") -> list[str]:
    """Get the plot names for the columns. 
       Prettier/more readable names for output. OMOP mapping is used for lab values."""
    kanta_omop_map = read_file(omop_mapping_path, schema={"measurement_concept_id": pl.Utf8, "concept_name": pl.Utf8})
    kanta_omop_map = dict(zip(kanta_omop_map["measurement_concept_id"].to_list(), kanta_omop_map["concept_name"].to_list()))
    new_names = []
    if lab_name == "hba1c": lab_name = "HbA1c"
    if lab_name == "egfr": lab_name = "eGFR"
    if lab_name == "ana": lab_name = "ANA"
    if lab_name == "tsh": lab_name = "TSH"
    if lab_name == "ldl": lab_name = "LDL"
    if lab_name_two == "ftri": lab_name_two == "Fasting TG"
    if lab_name_two == "t4": lab_name_two == "fT4"
    if lab_name_two == "cystc": lab_name_two == "Cystatin C"
    if lab_name_two == "fgluc": lab_name_two = "Fasting Glucose"
    for col_name in col_names:
        if col_name.startswith("S_"): 
            crnt_lab_name = lab_name_two
            col_name = col_name.replace("S_", "")
        else: 
            crnt_lab_name = lab_name
        if col_name.split("_")[0] in kanta_omop_map.keys():
            if "QUANT" in col_name:
                new_name = col_name.split("_")[1].replace("QUANT", "") + "% Quant - " + kanta_omop_map[col_name.split("_")[0]].split("[")[0]
                
            else:
                new_name = col_name.split("_")[1].capitalize() + " - " + kanta_omop_map[col_name.split("_")[0]].split("[")[0]
            if len(kanta_omop_map[col_name.split("_")[0]].split("]")) > 1:
                new_name += kanta_omop_map[col_name.split("_")[0]].split("]")[1]
        else:
            if col_name.startswith("QUANT"):
                new_name = col_name.split("_")[1] + "% Quant - " + crnt_lab_name
            elif col_name in ["MIN", "MEAN", "MAX", "MEDIAN", ]:
                new_name = col_name.capitalize() + " - " + crnt_lab_name
            elif col_name == "LAST_VAL_DIFF":
                new_name = "Distance of last value - " + crnt_lab_name
            elif col_name == "IDX_QUANT_100":
                new_name = "Last value - " + crnt_lab_name
            elif col_name == "IDX_QUANT_50":
                new_name = "Mid value - " + crnt_lab_name
            elif col_name == "IDX_QUANT_0":
                new_name = "First value - " + crnt_lab_name
            elif col_name == "EVENT_AGE":
                new_name = "Age"
            elif col_name == "FIRST_LAST":
                new_name = "Days between first and last - " + crnt_lab_name
            elif col_name == "SEX":
                new_name = "Sex"
            elif col_name == "MIN_LOC":
                new_name = "Location of min (-days) - " + crnt_lab_name
            elif col_name == "MAX_LOC":
                new_name = "Location of max (-days) - " + crnt_lab_name
            elif col_name == "REG_COEF":
                new_name = "Regression coef - " + crnt_lab_name
            elif col_name == "SUM_ABS_CHANGE":
                new_name = "Sum of absolute change - " + crnt_lab_name
            elif col_name == "MAX_ABS_CHANGE":
                new_name = "Max of absolute change - " + crnt_lab_name
            elif col_name == "MEAN_ABS_CHANGE":
                new_name = "Mean of absolute change - " + crnt_lab_name
            elif col_name == "ABNORM":
                new_name = "Number of prior abnorm - " + crnt_lab_name
            elif col_name == "MEAN_CHANGE":
                new_name = "Mean change - " + crnt_lab_name
            elif col_name == "MAX_CHANGE":
                new_name = "Max change - " + crnt_lab_name
            elif col_name == "SUM":
                new_name = "Sum - " + crnt_lab_name
            elif col_name == "SD":
                new_name = "Standard deviation - " + crnt_lab_name
            elif col_name == "ABS_ENERG":
                new_name = "Absolute energy - " + crnt_lab_name
            elif col_name == "SKEW":
                new_name = "Skew - " + crnt_lab_name
            elif col_name == "KURT":
                new_name = "Kurtosis - " + crnt_lab_name
            elif col_name == "SEQ_LEN":
                new_name = "Number of measurements - " + crnt_lab_name
            elif col_name == "SUM_CHANGE":
                new_name = "Sum of change - " + crnt_lab_name
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
    newcol = col_data.map_elements(lambda x: value_map.get(np.round(x), np.round(x)), return_dtype=pl.Float64)

    return(newcol, min_val, max_val)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Compound plots                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

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
