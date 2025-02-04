import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from plot_utils import *
import sklearn.metrics as skm
sys.path.append(("/home/ivm/valid/scripts/"))

def model_memory_size(clf):
    return sys.getsizeof(pickle.dumps(clf))

def get_train_type(metric):
    if metric == "tweedie": return("cont")
    if metric == "mse": return("cont")
    if metric == "logloss": return("bin")
    if metric == "F1" or metric == "f1": return("bin")

def get_score_func_based_on_metric(metric):
    if metric == "tweedie": return(skm.d2_tweedie_score)
    if metric == "mse": return(skm.mean_squared_error)
    if metric == "logloss": return(skm.log_loss)
    if metric == "F1" or metric == "f1": return(skm.f1_score)

def report(mdl, out_data, display_scores=[], 
           importance_plot=False, confusion_labels=None, feature_labels=None, verbose=True,
           metric="logloss"):
    """ Reports various metrics of the trained classifier """
    
    dump = dict()

    train_preds = out_data.query("SET==0")["ABNORM_PREDS"]
    valid_preds = out_data.query("SET==1")["ABNORM_PREDS"]

    if get_train_type(metric) == "bin":
        y_train = out_data.query("SET==0")["TRUE_VALUE"]
        y_valid = out_data.query("SET==1")["TRUE_VALUE"]
        y_probs = out_data.query("SET==1")["ABNORM_PROBS"]
        roc_auc = roc_auc_score(y_valid, y_probs)
    else:
        y_train = out_data.query("SET==0")["TRUE_ABNORM"]
        y_valid = out_data.query("SET==1")["TRUE_ABNORM"]

    train_acc = accuracy_score(y_train, train_preds)
    valid_acc = accuracy_score(y_valid, valid_preds)

    ## Additional scores
    scores_dict = dict()
    for score_name in display_scores:
        metric, func = get_abnorm_func_based_on_metric(metric)
        scores_dict[metric] = [func(y_train, train_preds), func(y_valid, valid_preds)]
        
    ## Model Memory
    model_mem = round(model_memory_size(mdl) / 1024, 2)
    
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
            logging_print(f'Valid {score_name}: ' + str(v[1]))
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
    
    if verbose:
        # logging_print("\n================================> CONFUSION MATRIX <=====================================")
    
        ## Confusion Matrix HeatMap
        logging_print("\n=======================================> PLOTS <=========================================")


        ## Variable importance plot
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
        conf_axes = axes[0, 0]
        roc_axes = axes[1, 0]
        pr_axes = axes[2, 0]
        importances = None

        if importance_plot:
            if not feature_labels:
                raise RuntimeError("'feature_labels' argument not passed when 'importance_plot' is True")

            try:
                importances = pd.Series(mdl.feature_importances_, index=feature_labels).sort_values(ascending=False)
                logging_print(importances)
            except AttributeError:
                try:
                    importances = pd.Series(mdl.coef_.ravel(), index=feature_labels).sort_values(ascending=False)
                except AttributeError:
                    try:
                        importances = pd.Series(mdl.get_score(importance_type="weight").values(), index=mdl.get_score(importance_type="weight").keys()).sort_values(ascending=False)
                    except AttributeError:
                        pass

            if importances is not None:
                # Modifying grid
                grid_spec = axes[0, 0].get_gridspec()
                for ax in axes[:, 0]:
                    ax.remove()   # remove first column axes
                large_axs = fig.add_subplot(grid_spec[0:, 0])

                # Plot importance curve
                feature_importance_plot(importances=importances.values,
                                        feature_labels=importances.index,
                                        ax=large_axs)
                large_axs.axvline(x=0)

                # Axis for ROC and PR curve
                conf_axes = axes[0, 1]
                roc_axes = axes[1, 1]
                pr_axes = axes[2, 1]

            else:
                # remove second row axes
                for ax in axes[:,1]: ax.remove()
        else:
            # remove second column axes
            for ax in axes[:,1]: ax.remove()

        confusion_plot(confusion_matrix(y_valid, valid_preds), labels=confusion_labels, ax=conf_axes)
        if get_train_type(metric) == "bin":
            ## ROC and Precision-Recall curves
            mdl_name = mdl.__class__.__name__
            roc_plot(y_valid, y_probs, mdl_name, ax=roc_axes)
            precision_recall_plot(y_valid, y_probs, mdl_name, ax=pr_axes)

            fig.subplots_adjust(wspace=5)
            fig.tight_layout()
        
    ## Dump to report_dict
    dump = dict(mdl=mdl, accuracy=[train_acc, valid_acc], **scores_dict,
                train_preds=train_preds,
                valid_preds=valid_preds,
                valid_probs=y_probs, report=mdl_rep, 
                roc_auc=roc_auc, model_memory=model_mem)
    
    return dump, fig

def compare_models(y_valid=None, mdl_reports=[], labels=[], score='accuracy'):
    """ Compare evaluation metrics for the True Positive class [1] of 
        binary classifiers passed in the argument and plot ROC and PR curves.
        
        Arguments:
        ---------
        y_valid: to plot ROC and Precision-Recall curves
         score: is the name corresponding to the sklearn metrics
        
        Returns:
        -------
        compare_table: pandas DataFrame containing evaluated metrics
                  fig: `matplotlib` figure object with ROC and PR curves """

    
    ## Classifier Labels
    default_names = [rep['clf'].__class__.__name__ for rep in mdl_reports]
    mdl_names = labels if len(labels) == len(mdl_reports) else default_names
    
    ## Compare Table
    table = dict()
    index = ['Train ' + score, 'Valid ' + score, 'Overfitting', 'ROC Area', 'Precision', 'Recall', 'F1-score', 'Support']
    for i in range(len(mdl_reports)):
        scores = [round(i, 3) for i in mdl_reports[i][score]]
        
        roc_auc = mdl_reports[i]['roc_auc']
        
        # Get metrics of True Positive class from sklearn classification_report
        true_positive_metrics = list(mdl_reports[i]['report']["1.0"].values())
        
        table[mdl_names[i]] = scores + [scores[1] < scores[0], roc_auc] + true_positive_metrics
    
    table = pd.DataFrame(data=table, index=index)
    
    
    ## Compare Plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # ROC and Precision-Recall
    for i in range(len(mdl_reports)):
        clf_probs = mdl_reports[i]['test_probs']
        roc_plot(y_valid, clf_probs, label=mdl_names[i], compare=True, ax=axes[0])
        precision_recall_plot(y_valid, clf_probs, label=mdl_names[i], compare=True, ax=axes[1])
    # Plot No-Info classifier
    axes[0].plot([0,1], [0,1], linestyle='--', color='green')
        
    fig.tight_layout()
    plt.close()
    
    return table.T, fig