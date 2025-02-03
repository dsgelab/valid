import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pretty_int(int_no):
    if int_no >= 1000000:
        return(str(round(int_no / 1000000,2)) + "M")
    elif int_no >= 1000:
        return(str(round(int_no / 1000)) + "K")
    else:
        return(str(round(int_no, 0)))
        
def round_column_min5(data, col):
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

def plot_time_score_metrics(time_scores, eval_metrics, N_indv, N_abnorm):
    min_y = min([min(time_scores.F1), min(time_scores.accuracy), min(time_scores.precision), min(time_scores.recall)])
    eval_all = eval_metrics.query("SET=='Valid' & SUBSET=='All'")
    fig, axes = plt.subplots(1, 3, figsize=(20,5))

    if eval_all.N_ABNORM.values[0] > 0:
        sns.lineplot(time_scores.loc[time_scores.SET=="Valid"][["SUBSET",  "F1", "tweedie", "accuracy"]].reset_index().melt(["index", "SUBSET"]), 
                     palette=["#D5694F", "#748AAA", "#CCB6AF"],
                     x="index", y="value", hue="variable", ax=axes[0])
        axes[0].set_ylim(min_y, 1)
        axes[0].set_xlabel("Years")
        axes[0].set_ylabel("")
        axes[0].set_title("Tweedie: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})  Accuracy: {:.2f} ({:.2f}-{:.2f})".format((round(eval_all.tweedie.values[0], 2)), (round(eval_all.tweedie_CIneg.values[0], 2)), (round(eval_all.tweedie_CIpos.values[0], 2)), (round(eval_all.F1.values[0], 2)), (round(eval_all.F1_CIneg.values[0], 2)), (round(eval_all.F1_CIpos.values[0], 2)), (round(eval_all.accuracy.values[0], 2)), (round(eval_all.accuracy_CIneg.values[0], 2)), (round(eval_all.accuracy_CIpos.values[0], 2)))) 
    
        sns.lineplot(time_scores.loc[time_scores.SET=="Valid"][["SUBSET",  "F1", "precision", "recall"]].reset_index().melt(["index", "SUBSET"]), 
                     palette=["#D5694F", "#748AAA", "#CCB6AF"],
                     x="index", y="value", hue="variable", ax=axes[1])
        axes[1].set_ylim(min_y, 1)
        axes[1].set_xlabel("Years")
        axes[1].set_label("")
        axes[1].set_title("Precision: {:.2f} ({:.2f}-{:.2f})  Recall: {:.2f} ({:.2f}-{:.2f})".format((round(eval_all.precision.values[0], 2)), (round(eval_all.precision_CIneg.values[0], 2)), (round(eval_all.precision_CIpos.values[0], 2)), (round(eval_all.recall.values[0], 2)), (round(eval_all.recall_CIneg.values[0], 2)), (round(eval_all.recall_CIpos.values[0], 2)))) 

    sns.lineplot(time_scores.loc[time_scores.SET=="Valid"][["SUBSET", "MSE"]].reset_index().melt(["index", "SUBSET"]), 
                 palette=["#841C26"], x="index", y="value", 
                 hue="variable", ax=axes[2])
    axes[2].set_xlabel("Years")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("MSE: {:.2f} ({:.2f}-{:.2f}): ".format((round(eval_all.MSE.values[0], 0)), (round(eval_all.MSE_CIneg.values[0], 0)), (round(eval_all.MSE_CIpos.values[0], 0)))) 

    return(fig)

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
        axes[0].set_title("Tweed: {:.2f} ({:.2f}-{:.2f})  F1: {:.2f} ({:.2f}-{:.2f})  Acc: {:.2f} ({:.2f}-{:.2f})".format(round(eval_all.tweedie.values[0], 2), round(eval_all.tweedie_CIneg.values[0], 2), round(eval_all.tweedie_CIpos.values[0], 2), round(eval_all.F1.values[0], 2), round(eval_all.F1_CIneg.values[0], 2), round(eval_all.F1_CIpos.values[0], 2), round(eval_all.accuracy.values[0], 2), round(eval_all.accuracy_CIneg.values[0], 2), round(eval_all.accuracy_CIpos.values[0], 2)))
        
        ## F1 and precision recall
        axes[1] = plot_melt_scores(scores, x_col_name, x_axis_name, ["F1", "precision", "recall"], axes[1], min_y)
        axes[1].text(55, 0.95, "N = " + pretty_int(eval_all.N_INDV.values[0]) +  "  A = " + pretty_int(eval_all.N_ABNORM.values[0]))
        axes[1].set_title("Precision: {:.2f} ({:.2f}-{:.2f})  Recall: {:.2f} ({:.2f}-{:.2f})".format(round(eval_all.precision.values[0], 2), round(eval_all.precision_CIneg.values[0], 2), round(eval_all.precision_CIpos.values[0], 2), (round(eval_all.recall.values[0], 2)), (round(eval_all.recall_CIneg.values[0], 2)), (round(eval_all.recall_CIpos.values[0], 2)))) 

    # MSE plot
    axes[2] = plot_melt_scores(scores, x_col_name, x_axis_name, ["MSE"], axes[2], min_y=np.nan, palette=["#841C26"])
    axes[2].set_title("MSE: {:.2f} ({:.2f}-{:.2f}): ".format((round(eval_all.MSE.values[0], 0)), (round(eval_all.MSE_CIneg.values[0], 0)), (round(eval_all.MSE_CIpos.values[0], 0)))) 
    
    return(fig)
    
def plot_observed_vs_probability_min5(data, col_name_x, col_name_y):
    sns.set_style('whitegrid')
    newcol_x, min_val_x, max_val_x = round_column_min5(data, col_name_x)
    if all(data[col_name_y] <= 1.0): data[col_name_y] = data[col_name_y]*100
    newcol_y, min_val_y, max_val_y = round_column_min5(data, col_name_y)

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=newcol_x, y=newcol_y, kind="hex", mincnt=5, cmap="twilight_shifted", marginal_kws=dict(stat="density", kde=True, discrete=True, color="#564D65"))
    g.ax_marg_y.set_ylim(0, 100)
    g.ax_joint.text(max(newcol_y)-(0.05*max(newcol_y)), 90, "N = " + pretty_int(data.shape[0]))

    g.ax_joint.set_xlabel("observed value")
    g.ax_joint.set_ylabel("probability of abnormality")
    return(g, newcol_x, newcol_y)
    
def plot_observerd_vs_predicted(data, col_name_x, col_name_y, col_name_x_abnorm):
    sns.set_style('whitegrid')
    axis_limits = [min(min(data[col_name_x]), min(data[col_name_y])), max(max(data[col_name_x]), max(data[col_name_y]))]
    
    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x=data[col_name_x], y=data[col_name_y], kind="scatter", hue=data[col_name_x_abnorm])
    g.ax_marg_x.set_xlim(axis_limits[0], axis_limits[1])
    g.ax_marg_y.set_ylim(axis_limits[0], axis_limits[1])
    g.ax_joint.set_xlabel("observed value")
    g.ax_joint.set_ylabel("predicted value")
    g.ax_joint.text(axis_limits[1]//4, axis_limits[1]-(0.1*axis_limits[1]), "N = " + pretty_int(data.shape[0]) +  "  A = " + pretty_int(data[col_name_x_abnorm].sum()))

    # # This is the x=y line using transforms
    g.ax_joint.plot(axis_limits, axis_limits, "#564D65", linestyle='dashdot', transform=g.ax_joint.transData)
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
    g.ax_joint.text(axis_limits[1]//4, axis_limits[1]-(0.1*axis_limits[1]), "N = " + pretty_int(data.shape[0]))
    
    # # This is the x=y line using transforms
    g.ax_joint.plot(axis_limits, axis_limits, "#564D65", linestyle='dashdot', transform=g.ax_joint.transData)
    return(g, newcol_x, newcol_y)