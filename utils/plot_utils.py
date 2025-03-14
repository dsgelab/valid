import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import shap
except:
    pass

sns.set_style('whitegrid')

def pretty_int(int_no):
    """Round a number to a pretty string with K (N>1000), M (N>1000000)."""
    if int_no >= 1000000:
        return(str(round(int_no / 1000000,2)) + "M")
    elif int_no >= 1000:
        return(str(round(int_no / 1000)) + "K")
    else:
        return(str(round(int_no, 0)))
        
def round_column_min5(data, col):
    """Rounds a column to the nearest 5 and replaces values with less than 5 counts with the nearest value with 5 counts.
        Returns the new column and the min and max values of the column."""
    mean_freqs = pd.DataFrame(data[col].round().value_counts()).reset_index()
    mean_freqs.columns = ["VALUE", "COUNT"]
    value_map = dict()
    min_val = mean_freqs[mean_freqs.COUNT >= 5].VALUE.min()
    max_val = mean_freqs[mean_freqs.COUNT >= 5].VALUE.max()
    for row_idx in np.arange(mean_freqs.shape[0]):
        if mean_freqs.iloc[row_idx].COUNT < 5:
            low_count_val = mean_freqs.iloc[row_idx].VALUE
            if abs(min_val-low_count_val) < abs(max_val-low_count_val):
                value_map[low_count_val] = min_val
            else:
                value_map[low_count_val] = max_val
    newcol = round(data[col]).map(value_map)
    newcol[newcol.isnull()] = round(data[col].loc[newcol.isnull()])
    return(newcol, min_val, max_val)
    
def get_plot_names(col_names, lab_name):
    kanta_omop_map = pd.read_csv("/home/ivm/valid/data/extra_data/upload/kanta_omop_mappings_counts.csv")
    kanta_omop_map = dict(zip(kanta_omop_map.measurement_concept_id.astype(str), kanta_omop_map.concept_name))
    new_names = []
    if lab_name == "hba1c": lab_name = "HbA1c"
    if lab_name == "egfr": lab_name == "eGFR"
        
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
            elif col_name == "EVENT_AGE":
                new_name = "Age"
            elif col_name == "FIRST_LAST":
                new_name = "Days between first and last - " + lab_name
            elif col_name == "SEX":
                new_name = "sex"
            elif col_name == "MIN_LOC":
                new_name = "Location of min (-days) - " + lab_name
            elif col_name == "MAX_LOC":
                new_name = "Location of max (-days) - " + lab_name
            elif col_name == "REG_COEF":
                new_name = "Regression coef - " + lab_name

                new_name = "sex"
            else:
                new_name = col_name

        new_names.append(new_name)
    return(new_names)
def plot_box_probs(evals=[], labels=[], ax1=None, ax2=None, ax_labels=["Controls", "Cases"]):
    axes = [ax1, ax2]
    for group in [0,1]:
        all_evals = pd.DataFrame()
        for idx, eval_x in enumerate(evals):
            eval_x = pd.DataFrame({"valid_probs": eval_x["valid_probs"], "y_valid": eval_x["y_valid"]})
            eval_crnt = eval_x.loc[eval_x.y_valid==group]
            safe = {"valid_probs": eval_crnt["valid_probs"].copy()*100}
            safe["valid_probs"], min_val, max_val = round_column_min5(safe, "valid_probs")
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

def create_shap_dir_plot(shap_values, plot_names):
    shap.summary_plot(shap_values, max_display=20, feature_names=plot_names, show=False)
    fig = plt.gcf()
    return fig

def create_report_plot(y_true, y_probs, y_preds, fg_down=False, train_type="bin", importance_plot=True, importances=None, confusion_labels=["Controls", "Cases"]):
    axes = None
    conf_axes = None
    roc_axes = None
    pr_axes = None
    cal_ax1 = None
    cal_ax2 = None
    imp_ax = None

    ## Variable importance plot
    axes = None
    if importance_plot and importances is not None:
        fig, (row_1, row_2, row_3) = plt.subplots(3, 2, figsize=(11, 10), gridspec_kw={'height_ratios': [1, 1, 0.5]})
        imp_ax, roc_axes = row_1[0], row_1[1]
        cal_ax1, pr_axes = row_2[0], row_2[1]
        cal_ax2, conf_axes = row_3[0], row_3[1]
        cal_ax1.sharex(cal_ax2)

        # Plot importance curve
        feature_importance_plot(importances=importances.feature_importance_vals, feature_labels=importances.feature_name, ax=imp_ax)

    if imp_ax is None: # remove second row axes
        fig, (row_1, row_2, row_3) = plt.subplots(3, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1, 0.25]})
        conf_axes, roc_axes = row_1[0], row_1[1]
        cal_ax1, pr_axes = row_2[0], row_2[1]
        cal_ax2, delete_axes = row_3[0], row_3[1]
        delete_axes.remove()
        cal_ax1.sharex(cal_ax2)
            
    confusion_plot(confusion_matrix(y_true, y_preds), labels=confusion_labels, ax=conf_axes)
    if train_type == "bin":
        ## ROC and Precision-Recall curves
        #mdl_name = mdl.__class__.__name__
        roc_plot(y_true, y_probs, "Test", ax=roc_axes)
        precision_recall_plot(y_true, y_probs, "Test", ax=pr_axes)
        plot_calibration(y_true, y_probs, cal_ax1, cal_ax2, fg_down=fg_down)
        fig.subplots_adjust(wspace=5)
        fig.tight_layout()
    return fig

"""Reshapes scores dataframe so that each metric gets its own column and then plots lines based on this."""
def plot_melt_scores(scores, x_col_name, x_axis_name, metrics, axis, min_y, palette=["#D5694F", "#748AAA", "#CCB6AF"]):
    plt_data = scores.loc[scores.SET=="Valid"][[x_col_name, *metrics]].reset_index(names="YEAR").melt(["YEAR", x_col_name])

    if x_axis_name == "Age": sns.lineplot(x=plt_data[x_col_name], y=plt_data["value"], hue=plt_data["variable"], palette=palette, ax=axis)
    elif x_axis_name == "Years": sns.lineplot(x=plt_data.YEAR, y=plt_data["value"], hue=plt_data["variable"], palette=palette, ax=axis)

    if not np.isnan(min_y): axis.set_ylim(min_y, 1)
    axis.set_xlabel(x_axis_name)
    axis.set_ylabel("")
    return(axis)
        
def plot_scores_metrics(scores, eval_metrics, x_axis_name, x_col_name="AGE_MID", subset_name="all"):
    ## Preparing
    eval_all = eval_metrics.loc[np.logical_and(np.logical_and(eval_metrics.SET=="Valid", eval_metrics.SUBSET == subset_name), eval_metrics.GROUP == "all")]
    fig, axes = plt.subplots(1, 3, figsize=(20,5))
    min_y = min([min(scores.F1), min(scores.accuracy), min(scores.precision), min(scores.recall), min(scores.tweedie)])

    ## Making sure there is enough data for the second two plots
    if eval_all.N_ABNORM.values[0] > 0:
        axes[0] = plot_melt_scores(scores, x_col_name, x_axis_name, ["F1", "tweedie", "accuracy"], axes[0], min_y)
        axes[0].legend(title="Score", loc="lower left")
        axes[0].text(0.72, 0.05, f'Tweedie = { round(eval_all.tweedie.values[0]) } \n Accuracy = {round(eval_all.accuracy.values[0])} \n F1 = {round(eval_all.F1.values[0])}', fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
        
        ## F1 and precision recall
        axes[1] = plot_melt_scores(scores, x_col_name, x_axis_name, ["F1", "precision", "recall"], axes[1], min_y)
        axes[1].legend(title="Score", loc="lower left")
        axes[1].text(55, 0.95, "N = " + pretty_int(eval_all.N_INDV.values[0]) +  "  A = " + pretty_int(eval_all.N_ABNORM.values[0]), fontsize=12, bbox=dict(facecolor='red', alpha=0.4, pad=5))
        axes[1].text(75, 0.05, f'Precision = { round(eval_all.precision.values[0, 2]) } \n Recall = {round(eval_all.recall.values[0], 2)}', fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))


    # MSE plot
    axes[2] = plot_melt_scores(scores, x_col_name, x_axis_name, ["MSE"], axes[2], min_y=np.nan, palette=["#841C26"])
    axes[2].set_title("MSE: {:.2f} ({:.2f}-{:.2f}): ".format((round(eval_all.MSE.values[0], 0)), (round(eval_all.MSE_CIneg.values[0], 0)), (round(eval_all.MSE_CIpos.values[0], 0)))) 
    axes[2].legend.set_visible(False)
    return(fig)
    
def plot_observed_vs_probability_min5(data, col_name_x, col_name_y):
    sns.set_style('whitegrid')
    data_temp = data.copy()
    newcol_x, min_val_x, max_val_x = round_column_min5(data_temp, col_name_x)
    if all(data_temp[col_name_y] <= 1.0): data_temp[col_name_y] = data_temp[col_name_y]*100
    newcol_y, min_val_y, max_val_y = round_column_min5(data_temp, col_name_y)

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=newcol_x, y=newcol_y, kind="hex", mincnt=5, cmap="twilight_shifted", marginal_kws=dict(stat="density", kde=True, discrete=True, color="#564D65"))
    g.ax_joint.text(max_val_x-(0.2*max_val_x), 90, "N = " + pretty_int(data.shape[0]), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))
    g.ax_marg_y.set_ylim(0, 100)
    g.ax_joint.set_xlabel("Observed Value")
    g.ax_joint.set_ylabel("Predicted Probability")
    return(g, newcol_x, newcol_y)

def plot_observerd_vs_predicted(data, col_name_x, col_name_y, col_name_x_abnorm, prob=True):
    sns.set_style('whitegrid')
    axis_limits = [min(min(data[col_name_x]), min(data[col_name_y])), max(max(data[col_name_x]), max(data[col_name_y]))]
    
    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=data[col_name_x], y=data[col_name_y]*100, kind="scatter", hue=data[col_name_x_abnorm])
    g.ax_marg_x.set_xlim(axis_limits[0], axis_limits[1])
    g.ax_joint.set_xlabel("Observed Value")
    if prob: 
        g.ax_joint.set_ylabel("Predicted Probability")
        g.ax_marg_y.set_ylim(0, 100)
        g.ax_joint.text(axis_limits[1]//2, 75, "N = " + pretty_int(data.shape[0]) +  "  A = " + pretty_int(data[col_name_x_abnorm].sum()), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))

    else: 
        g.ax_joint.set_ylabel("Predicted Value")
        g.ax_marg_y.set_ylim(axis_limits[0], axis_limits[1]) 
        # # This is the x=y line using transforms
        g.ax_joint.plot(axis_limits, axis_limits, "#564D65", linestyle='dashdot', transform=g.ax_joint.transData)
        g.ax_joint.text(axis_limits[1]//4, axis_limits[1]-(0.2*axis_limits[1]), "N = " + pretty_int(data.shape[0]) +  "  A = " + pretty_int(data[col_name_x_abnorm].sum()), fontsize=12, bbox=dict(facecolor='green', alpha=0.4, pad=5))

    return(g)
    
def plot_observerd_vs_predicted_min5(data, col_name_x, col_name_y):
    sns.set_style('whitegrid')

    newcol_x, min_val_x, max_val_x = round_column_min5(data, col_name_x)
    newcol_y, min_val_y, max_val_y = round_column_min5(data, col_name_y)
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

###################### Plotting from kaggel https://www.kaggle.com/code/para24/xgboost-stepwise-tuning-using-optuna#3.---Utility-Functions ######################
import timeit
import pickle
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, accuracy_score
from sklearn.exceptions import NotFittedError

def confusion_plot(matrix, labels=None, ax=None):
    """ Display binary confusion matrix as a Seaborn heatmap """
    
    labels = labels if labels else ['Control', 'Case']
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1)
    sns.heatmap(data=matrix, cmap='Blues', annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axis, cbar=False, square=True)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('Actual')
    axis.set_title('Confusion Matrix')
    return axis if ax else fig

def roc_plot(y_true, y_probs, label, compare=False, ax=None):
    """ Plot Receiver Operating Characteristic (ROC) curve 
        Set `compare=True` to use this function to compare classifiers. """
    
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

import scipy
def round_column_min5(data, col):
    """Rounds a column to the nearest 5 and replaces values with less than 5 counts with the nearest value with 5 counts.
        Returns the new column and the min and max values of the column."""
    mean_freqs = pd.DataFrame(data[col].round().value_counts()).reset_index()
    mean_freqs.columns = ["VALUE", "COUNT"]
    value_map = dict()
    min_val = mean_freqs[mean_freqs.COUNT >= 5].VALUE.min()
    max_val = mean_freqs[mean_freqs.COUNT >= 5].VALUE.max()
    for row_idx in np.arange(mean_freqs.shape[0]):
        if mean_freqs.iloc[row_idx].COUNT < 5:
            low_count_val = mean_freqs.iloc[row_idx].VALUE
            if abs(min_val-low_count_val) < abs(max_val-low_count_val):
                value_map[low_count_val] = min_val
            else:
                value_map[low_count_val] = max_val
    newcol = round(data[col]).map(value_map)
    newcol[newcol.isnull()] = round(data[col].loc[newcol.isnull()])
    return(newcol, min_val, max_val)
    
def plot_calibration(y_true, y_probs, ax1=None, ax2=None, bin_type="equal", compare=False, label="Calibration", fg_down=True):
    if ax1 is None:  fig, (ax1,ax2) = plt.subplots(2,figsize=(15,7), sharex=True, sharey=False, height_ratios=[1, .5])
    if bin_type != "equal": bins = np.quantile(y_probs, np.arange(0,1.1,0.1))
    else: bins=np.arange(0, 1.1, 0.1)
    print(bins)
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
            safe_cases = {"valid_probs": y_probs[y_true==1].copy()*100}
            safe_cases, _, _ = round_column_min5(safe_cases, "valid_probs")
            safe_controls = {"valid_probs": y_probs[y_true==0].copy()*100}
            safe_controls, _, _ = round_column_min5(safe_controls, "valid_probs")
            safe = pd.concat([safe_cases, safe_controls])
            labels = pd.concat([y_true[y_true==1], y_true[y_true==0]])
        else:
            safe = y_probs*100
            labels = y_true
        sns.boxplot(x=safe/100, 
                    y=pd.Categorical(labels.map({0: "Control", 1: "Case"}), categories=["Control", "Case"]), 
                    ax=ax2, color="black", vert=False, width=.5, whis=(5,95), fill=False, 
                    flierprops=dict(marker=".", markeredgecolor="white", markerfacecolor="k"))
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('')
        
import sklearn.metrics as skm   
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

def feature_importance_plot(importances, feature_labels, ax=None, n_import=10):
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x=importances[0:n_import], y=feature_labels[0:n_import], hue=feature_labels[0:n_import], ax=axis)
    axis.set_title('Feature Importances')
    axis.set_ylabel("")
    axis.set_xlabel("mean(|SHAP value|)")

    plt.close()
    
    return axis if ax else fig

