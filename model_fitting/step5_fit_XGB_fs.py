# Utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from code.valid.utils.optuna_utils import run_optuna_optim_cv
from model_eval_utils import bootstrap_difference
from general_utils import get_date, make_dir, init_logging, Timer, logging_print
from model_eval_utils import get_train_type, save_all_report_plots
from optuna_utils import run_optuna_optim
from xgb_utils import create_xgb_dts, get_shap_importances, save_importances, get_out_data
from input_utils import get_data_and_pred_list   
from model_fit_utils import xgb_final_fitting, get_xgb_base_params
from labeling_utils import log_print_n

# Needed for metric comparison
from model_eval_utils import bootstrap_difference
import sklearn.metrics as skm
from delong_utils import delong_roc_test

# Standard stuff
import polars as pl
import xgboost as xgb
import pickle
import shap

# Logging and input
import argparse
import logging
logger = logging.getLogger(__name__)


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--model_fit_date", type=str, help="Original date of model fitting.", default="")
    parser.add_argument("--fg_ver", type=str, help="FinnGen version needed for import of BMI or SMOKE.", default="r13")

    # Data paths
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument("--file_path_icds", type=str, help="Path to ICD data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_atcs", type=str, help="Path to ATC data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_labs", type=str, help="Path to Lab data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_sumstats", type=str, help="Path to summary statistics of a single lab value data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_second_sumstats", type=str, help="Path to summary statistics of a another lab value data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_pgs1", type=str, help="PGS scores - 1/2", default="")
    parser.add_argument("--file_path_pgs2", type=str, help="PGS scores - 2/2", default="")

    # Extra info
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--lab_name_two", type=str, help="Readable name of the second most relevant measurement value for file naming.", default="")
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before")

    # Prediction task
    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--abnorm_extra_choice", default="", type=str, help="Extra choice for abnormality prediction. [default: ''] ")
    parser.add_argument("--preds", type=str, help="List of predictors. Special options: ICD_MAT, ATC_MAT, SUMSTATS, SECOND_SUMSTATS, LAB_MAT. Taking all columns from the respective data file.", 
                        default=["SUMSTATS", "EVENT_AGE", "SEX"], nargs="+")
    
    # Model fitting parameters
    parser.add_argument("--model_type", type=str, help="XGBoost or elastic net. [default: 'xgb'] options: xgb or elr ", default="xgb")

    parser.add_argument("--lr", type=float, help="Learning rate for hyperparamter optimization, can be high so this goes fast.", default=0.4)
    parser.add_argument("--low_lr", type=float, help="Learning rate for final model training.", default=0.001)
    parser.add_argument("--early_stop", type=int, help="Early stopping for the final fitting round. Currently, early stopping fixed at 5 for hyperparameter optimization.", default=5)
    parser.add_argument("--metric", type=str, help="Which metric to optimize based on.", default="mse")

    # Hyperparameter optimization parameters
    parser.add_argument("--n_trials", type=int, help="Number of hyperparameter optimizations to run [default: 1 = running based on time_step1 instead]", default=1)
    parser.add_argument("--time_optim", type=int, help="Number of seconds to run hyperparameter optimizations for, instead of basing it on the number of traisl. [run when n_trials=1]", default=300)
    parser.add_argument("--refit", type=int, help="Whether to rerun the hyperparameter optimization", default=1)

    # Final model fitting and evaluation
    parser.add_argument("--skip_model_fit", type=int, help="Whether to rerun the final model fitting, or load a prior model fit.", default=0)
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics.", default=500)
    parser.add_argument("--fit_cv", type=int, help="Do final fit on all of training with cross-validation. Otherwise fit only on training with early stopping on validation.", default=500)
    parser.add_argument("--final_fit", type=int, help="Do one final fit on all data.", default=0)
    parser.add_argument("--fids_path", type=str, help="IDs to filter data for.", default="")
    parser.add_argument("--filter_name", type=str, help="Name of ID filter.", default="")

    # Feature selection parameters
    parser.add_argument("--n_features", type=int, help="Number of features to select.", default=0)
    parser.add_argument("--fs_path", type=str, help="Path to feature selection results.", default="")
    parser.add_argument("--train_pct", type=float, help="Amount of training data to be used", default=100)

    args = parser.parse_args()

    return(args)

def eval_metric_diff(full_out_data,
                  out_data: pl.DataFrame,
                  set_no: int,
                  crnt_k: int,
                  fs_results: pl.DataFrame,
                  metric: str="auc"):
    crnt_preds = (full_out_data
                    .filter(pl.col.SET==set_no)
                    .select(["TRUE_ABNORM", "ABNORM_PROBS", "FINNGENID"])
                    .join(out_data.select(["FINNGENID", "ABNORM_PROBS"]), on="FINNGENID", how="inner")
                )
    if metric == "auc":
        metric_func = skm.roc_auc_score
    elif metric == "avg_prec":
        metric_func = skm.average_precision_score
    elif metric == "logloss":
        metric_func = skm.log_loss
    diff_est, lowci, highci, pval_diff, avg_1, avg_2 = bootstrap_difference(metric_func = metric_func,
                                                                    preds_1=crnt_preds["ABNORM_PROBS"].to_numpy(),
                                                                    preds_2=crnt_preds["ABNORM_PROBS_right"].to_numpy(),
                                                                    obs=crnt_preds["TRUE_ABNORM"].to_numpy(),
                                                                    n_boots=500)
    if metric == "auc":
        pval_diff = 10**delong_roc_test(crnt_preds["TRUE_ABNORM"].to_numpy(), crnt_preds["ABNORM_PROBS"].to_numpy(), crnt_preds["ABNORM_PROBS_right"].to_numpy())[0][0]

    fs_results_dict = {"SET": set_no, 
                       "METRIC": metric,
                         "N_FEATURES": crnt_k, 
                         "PVAL_DIFF": pval_diff, 
                         "DIFF_EST": diff_est, 
                         "LOW_CI": lowci, 
                         "HIGH_CI": highci,
                         "AVG_FULL": avg_1,
                         "AVG_FS": avg_2}
    temp_fs_results = (pl.DataFrame(fs_results_dict)
                       .with_columns(pl.col.SET.cast(pl.Float32),
                                     pl.col.METRIC.cast(pl.Utf8),
                                     pl.col.N_FEATURES.cast(pl.Int32),
                                        pl.col.PVAL_DIFF.cast(pl.Float64),
                                        pl.col.DIFF_EST.cast(pl.Float64),
                                        pl.col.LOW_CI.cast(pl.Float64),
                                        pl.col.HIGH_CI.cast(pl.Float64),
                                        pl.col.AVG_FULL.cast(pl.Float64),
                                        pl.col.AVG_FS.cast(pl.Float64)
                       )
                    )
    
    fs_results = pl.concat([fs_results, temp_fs_results])
    return fs_results, pval_diff, diff_est, lowci, highci, avg_1, avg_2



if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    timer = Timer()
    args = get_parser_arguments()

    # File names
    study_name = args.model_type + "_" + str(args.metric) + "_" + args.pred_descriptor 
    if args.filter_name != "": study_name + "_" + args.filter_name
    if args.model_fit_date == "": args.model_fit_date = get_date()

    out_dir = args.res_dir + study_name + "/";    
    if args.train_pct<100: out_dir = out_dir + f"trainpct{args.train_pct}/" 
    if args.final_fit: out_dir = out_dir + "/final/"
    out_model_dir = out_dir + "models/" + args.lab_name + "/" + args.model_fit_date + "/" 
    out_plot_dir = out_dir + "plots/" + args.model_fit_date + "/"
    out_plot_path = out_plot_dir + args.lab_name 
    out_down_dir = out_dir + "down/" + args.model_fit_date + "/" 
    out_down_path = out_down_dir + args.lab_name 

    init_logging(out_dir, args.lab_name, logger, args)
    make_dir(out_model_dir); make_dir(out_plot_dir); make_dir(out_down_dir)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data, X_cols = get_data_and_pred_list(file_path_labels=args.file_path_labels, 
                                          file_path_icds=args.file_path_icds, 
                                          file_path_atcs=args.file_path_atcs, 
                                          file_path_sumstats=args.file_path_sumstats, 
                                          file_path_second_sumstats=args.file_path_second_sumstats, 
                                          file_path_labs=args.file_path_labs, 
                                          file_path_pgs1=args.file_path_pgs1,
                                          file_path_pgs2=args.file_path_pgs2,
                                          preds=args.preds,
                                          start_date=args.start_date,
                                          fill_missing=0 if args.model_type=="xgb" else 1,
                                          fids_path=args.fids_path)

    fgids, X_train, y_train, \
        X_finetune_valid, y_finetune_valid, \
        X_valid, y_valid, \
        X_test, y_test, \
        X_all, y_all, X_all_unscaled, \
        dtrain, dfinetunevalid, dvalid, \
            scaler_base, data = create_xgb_dts(data=data, 
                                               X_cols=X_cols, 
                                               y_goal=args.goal,
                                               train_pct=args.train_pct)
    print(X_all_unscaled.with_columns(pl.Series("FINNGENID", fgids)).head(2))
    log_print_n(data.filter(pl.col.SET==0), "Train")
    log_print_n(data.filter(pl.col.SET==0.5), "Finetune Valid")
    log_print_n(data.filter(pl.col.SET==1), "Valid")
    log_print_n(data.filter(pl.col.SET==2), "Test")
    print(data["SET"].value_counts(normalize=True))
    X_all_unscaled.with_columns(pl.Series("FINNGENID", fgids)).write_parquet(out_model_dir + "Xall_unscaled_" + get_date() + ".parquet")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Hyperparam optimization with optuna                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    base_params = get_xgb_base_params(metric=args.metric, 
                                          lr=args.lr, 
                                          n_classes=len(y_train.unique()))
    best_params = run_optuna_optim_cv(train=[pl.concat([X_train, X_finetune_valid]), pl.concat([y_train, y_finetune_valid])], 
                                      lab_name=args.lab_name, 
                                      refit=args.refit, 
                                      time_optim=args.time_optim, 
                                      n_trials=args.n_trials, 
                                      study_name=study_name,
                                      res_dir=args.res_dir,
                                      model_type="xgb",
                                      model_fit_date=args.model_fit_date,
                                      base_params=base_params)
    if not args.skip_model_fit:
        logging.info(timer.get_elapsed())
        full_model = xgb_final_fitting(best_params=best_params,
                                             X_train=X_train, y_train=y_train, 
                                             X_finetune_valid=X_finetune_valid, 
                                             y_finetune_valid=y_finetune_valid, 
                                             X_valid=X_valid, y_valid=y_valid,
                                             X_test=X_test, y_test=y_test, 
                                             metric=args.metric,
                                             low_lr=args.low_lr,
                                             early_stop=args.early_stop,
                                             n_classes=len(y_train.unique()),
                                             fit_cv=False,
                                             final_fit=args.final_fit)
        full_model_cv = xgb_final_fitting(best_params=best_params,
                                                X_train=X_train, y_train=y_train, 
                                                X_finetune_valid=X_finetune_valid, 
                                                y_finetune_valid=y_finetune_valid, 
                                                X_valid=X_valid, y_valid=y_valid,
                                                X_test=X_test, y_test=y_test, 
                                                metric=args.metric,
                                                low_lr=args.low_lr,
                                                early_stop=args.early_stop,
                                                n_classes=len(y_train.unique()),
                                                fit_cv=True,
                                                final_fit=args.final_fit)
        full_model_cv.save_model(out_model_dir + "cv_model_" + get_date() + ".pkl")  
        full_model.save_model(out_model_dir + "model_" + get_date() + ".pkl")  

        # # # # Shaps
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        shap_explainer = shap.TreeExplainer(full_model)
        train_importances, _ = get_shap_importances(X_train, 
                                                       shap_explainer, 
                                                       args.lab_name, 
                                                       args.lab_name_two, 
                                                       translate=True)
        shap_explainer = shap.TreeExplainer(full_model_cv)
        train_cv_importances, _ = get_shap_importances(pl.concat([X_train, X_finetune_valid]), 
                                                       shap_explainer, 
                                                       args.lab_name, 
                                                       args.lab_name_two, 
                                                       translate=True)

        valid_importances, _ = get_shap_importances(X_valid, 
                                                    shap_explainer, 
                                                    args.lab_name, 
                                                    args.lab_name_two, 
                                                    translate=True)
        test_importances, _ = get_shap_importances(X_test, 
                                                   shap_explainer, 
                                                   args.lab_name, 
                                                   args.lab_name_two, 
                                                   translate=True)
        save_importances(top_gain=train_importances,
                         out_down_path=out_down_path,
                         study_name=study_name,
                         lab_name=args.lab_name,
                         goal=args.goal,
                         subset="train")
        save_importances(top_gain=train_cv_importances,
                         out_down_path=out_down_path,
                         study_name=study_name,
                         lab_name=args.lab_name,
                         goal=args.goal,
                         subset="train",
                         cv=True)
        
        # # # # Model predictions
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        full_cv_data = get_out_data(data=data, 
                                     model_final=full_model_cv, 
                                     X_all=X_all, 
                                     y_all=y_all, 
                                     metric=args.metric,
                                     lab_name=args.lab_name,
                                     goal=args.goal,
                                     abnorm_extra_choice=args.abnorm_extra_choice)
        full_cv_data.write_parquet(out_model_dir + "cv_preds_" + get_date() + ".parquet")      

        full_data = get_out_data(data=data, 
                                     model_final=full_model, 
                                     X_all=X_all, 
                                     y_all=y_all, 
                                     metric=args.metric,
                                     lab_name=args.lab_name,
                                     goal=args.goal,
                                     abnorm_extra_choice=args.abnorm_extra_choice)
        full_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")          

        # # # # Plotting
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        save_all_report_plots(out_data=full_cv_data,
                                  out_plot_path=out_plot_path,
                                  out_down_path=out_down_path,
                                  study_name=study_name,
                                  train_importances=train_importances,
                                  valid_importances=valid_importances,
                                  test_importances=test_importances,
                                  train_type=get_train_type(args.metric),
                                  model_type=args.model_type,
                                  fit_cv=args.fit_cv)
    else:
        full_data = pl.read_parquet(out_model_dir + "preds_" + get_date() + ".parquet")
        train_importances = pl.read_csv(out_down_path + "/" + args.goal + "/" + args.lab_name + "_" + study_name + "_" + args.goal + "_shap_importance_train_" + get_date() + ".csv")
        full_model = xgb.XGBClassifier()
        full_model.load_model(out_model_dir + "model_" + get_date() + ".pkl")  

    # # # # Top K features
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # get top K features
    if args.n_features or args.fs_path != "":
        if args.n_features  :
            fs_results = pl.DataFrame()
            for crnt_k in range(1, args.n_features + 1):
                logging_print(f"Fitting feature select model with top {crnt_k} features.")
                # Top k without age and

                # Top k without age and sex
                top_k_features = (train_importances
                                        .filter(~pl.col.orig.is_in(["EVENT_AGE", "SEX"]))
                                        .select("orig")
                                        .head(crnt_k)
                                        .to_series()
                                        .to_list()
                                        )
                # Add age+sex 
                top_k_features = ["EVENT_AGE", "SEX"] + top_k_features

                X_fs_train = X_train.select(top_k_features)
                X_fs_finetune_valid = X_finetune_valid.select(top_k_features)
                X_fs_valid = X_valid.select(top_k_features)
                X_fs_test = X_test.select(top_k_features)
                X_fs_all = X_all.select(top_k_features)

                best_fs_params = run_optuna_optim_cv(train=[X_fs_train, y_train], 
                                                lab_name=args.lab_name, 
                                                refit=args.refit, 
                                                time_optim=args.time_optim, 
                                                n_trials=args.n_trials, 
                                                study_name=study_name+"_fs" + str(crnt_k),
                                                res_dir=args.res_dir,
                                                model_type="xgb",
                                                model_fit_date=args.model_fit_date,
                                                base_params=base_params)
                
                model_fs = xgb_final_fitting(best_params=best_fs_params,
                                                X_train=X_fs_train, y_train=y_train, 
                                                X_finetune_valid=X_fs_finetune_valid, y_finetune_valid=y_finetune_valid,
                                                X_valid=X_fs_valid, y_valid=y_valid, 
                                                X_test=X_fs_test, y_test=y_test, 
                                                metric=args.metric,
                                                low_lr=args.low_lr,
                                                early_stop=args.early_stop,
                                                n_classes=len(y_train.unique()),
                                                fit_cv=False,
                                                final_fit=args.final_fit)
                        
                fs_data = get_out_data(data=data, 
                                        model_final=model_fs, 
                                        X_all=X_fs_all, 
                                        y_all=y_all, 
                                        metric=args.metric,
                                        lab_name=args.lab_name,
                                        goal=args.goal,
                                        abnorm_extra_choice=args.abnorm_extra_choice)
                # # # # Check significance against full model
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                for crnt_metric in ["auc", "avg_prec", "logloss"]:
                    fs_results, _, _, _, _, _, _ = eval_metric_diff(full_out_data=full_data,
                                                                    out_data=fs_data,
                                                                    set_no=1,
                                                                    crnt_k=crnt_k,
                                                                    fs_results=fs_results,
                                                                    metric=crnt_metric)
                    fs_results, _, _, _, _, _, _ = eval_metric_diff(full_out_data=full_data,
                                                                    out_data=fs_data,
                                                                    set_no=2,
                                                                    crnt_k=crnt_k,
                                                                    fs_results=fs_results,
                                                                    metric=crnt_metric)

                fs_results, pval_diff, diff_est, lowci, highci, avg_low_1, avg_high_1 = eval_metric_diff(full_out_data=full_data,
                                                                                                     out_data=fs_data,
                                                                                                     set_no=0.5,
                                                                                                     crnt_k=crnt_k,
                                                                                                     fs_results=fs_results,
                                                                                                     metric="auc")
                fs_results, pval_diff_2, diff_est_2, lowci_2, highci_2, avg_low_2, avg_high_2 = eval_metric_diff(full_out_data=full_data,
                                                                                                     out_data=fs_data,
                                                                                                     set_no=0.5,
                                                                                                     crnt_k=crnt_k,
                                                                                                     fs_results=fs_results,
                                                                                                     metric="avg_prec")
                fs_results, pval_diff_3, diff_est_3, lowci_3, highci_3, avg_low_3, avg_high_3 = eval_metric_diff(full_out_data=full_data,
                                                                                                     out_data=fs_data,
                                                                                                     set_no=0.5,
                                                                                                     crnt_k=crnt_k,
                                                                                                     fs_results=fs_results,
                                                                                                     metric="logloss")
                print(fs_results)
                if ((pval_diff >= 0.05 and diff_est > -0.005) or (pval_diff <0.05 and diff_est > 0)) and \
                        ((pval_diff_2 >= 0.05 and diff_est_2 > -0.005) or (pval_diff_2 < 0.05 and diff_est_2 > 0)) and \
                        ((pval_diff_3 >= 0.05 and diff_est_3 < 0.005) or (pval_diff_3 < 0.05 and diff_est_3 < 0)):
                                                    # # # # Logging results
                    print(f"Selected top {crnt_k} features model as final FS model. {crnt_k} \
                            \nfeatures AUC p-value difference to full model: {pval_diff:.4f}, \
                            \nAUC diff estimate: {diff_est:.4f} (CI: {lowci:.4f}-{highci:.4f}), \
                            \nFull model AUC: {avg_low_1:.4f}, FS model AUC: {avg_high_1:.4f} \
                            \n\nAvgPrec p-value difference to full model: {pval_diff_2:.4f}, \
                            \n AvgPrec diff estimate: {diff_est_2:.4f} (CI: {lowci_2:.4f}-{highci_2:.4f}), \
                            \nFull model AvgPrec: {avg_low_2:.4f}, FS model AvgPrec: {avg_high_2:.4f} \
                            \n\nLogloss p-value difference to full model: {pval_diff_3:.4f}, \
                            \n Logloss diff estimate: {diff_est_3:.4f} (CI: {lowci_3:.4f}-{highci_3:.4f}), \
                            \nFull model Logloss: {avg_low_3:.4f}, FS model Logloss: {avg_high_3:.4f} \
                            ")
                    fs_results.write_csv(out_model_dir + "fs_auc_diffs_" + get_date() + ".tsv", separator="\t")
                    args.n_features = crnt_k
                    X_fs_all.with_columns(pl.Series("FINNGENID", fgids)).write_parquet(out_model_dir + "Xall_unscaled_" + "fs_"+ str(args.n_features) + "_" + get_date() + ".parquet")

                    break
                                            
        elif args.fs_path != "":
            fs_select = pl.read_csv(args.fs_path)
            args.n_features = fs_select.height
            top_k_features = (train_importances
                                .join(fs_select, on="orig", how="inner")
                                .select("orig")
                                .to_series()
                                .to_list()
            )


            X_fs_train = X_train.select(top_k_features)
            X_fs_finetune_valid = X_finetune_valid.select(top_k_features)
            X_fs_valid = X_valid.select(top_k_features)
            X_fs_test = X_test.select(top_k_features)
            X_fs_all = X_all.select(top_k_features)
            best_fs_params = run_optuna_optim_cv(train=[pl.concat([X_fs_train, X_fs_finetune_valid]), pl.concat([y_train, y_finetune_valid])], 
                                                lab_name=args.lab_name, 
                                                refit=args.refit, 
                                                time_optim=args.time_optim, 
                                                n_trials=args.n_trials, 
                                                study_name=study_name,
                                                res_dir=args.res_dir,
                                                model_type="xgb",
                                                model_fit_date=args.model_fit_date,
                                                base_params=base_params)
            model_fs = xgb_final_fitting(best_params=best_fs_params,
                                            X_train=X_fs_train, y_train=y_train, 
                                            X_finetune_valid=X_fs_finetune_valid, y_finetune_valid=y_finetune_valid,
                                            X_valid=X_fs_valid, y_valid=y_valid, 
                                            X_test=X_fs_test, y_test=y_test, 
                                            metric=args.metric,
                                            low_lr=args.low_lr,
                                            early_stop=args.early_stop,
                                            n_classes=len(y_train.unique()),
                                            fit_cv=False,
                                            final_fit=args.final_fit)
                        
            fs_data = get_out_data(data=data, 
                                    model_final=model_fs, 
                                    X_all=X_fs_all, 
                                    y_all=y_all, 
                                    metric=args.metric,
                                    lab_name=args.lab_name,
                                    goal=args.goal,
                                    abnorm_extra_choice=args.abnorm_extra_choice)
            # # # # Check significance against full model
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            fs_results = pl.DataFrame()
            for crnt_metric in ["auc", "avg_prec", "logloss"]:
                fs_results, _, _, _, _, _, _ = eval_metric_diff(full_out_data=full_data,
                                                                out_data=fs_data,
                                                                set_no=0,
                                                                crnt_k=args.n_features,
                                                                fs_results=fs_results,
                                                                metric=crnt_metric)
                fs_results, _, _, _, _, _, _ = eval_metric_diff(full_out_data=full_data,
                                                                out_data=fs_data,
                                                                set_no=1,
                                                                crnt_k=args.n_features,
                                                                fs_results=fs_results,
                                                                metric=crnt_metric)
                fs_results, _, _, _, _, _, _ = eval_metric_diff(full_out_data=full_data,
                                                                out_data=fs_data,
                                                                set_no=2,
                                                                crnt_k=args.n_features,
                                                                fs_results=fs_results,
                                                                metric=crnt_metric)
                
            fs_results, pval_diff, diff_est, lowci, highci, avg_low_1, avg_high_1 = eval_metric_diff(full_out_data=full_data,
                                                                                                     out_data=fs_data,
                                                                                                     set_no=0.5,
                                                                                                     crnt_k=args.n_features,
                                                                                                     fs_results=fs_results,
                                                                                                     metric="auc")
            fs_results, pval_diff_2, diff_est_2, lowci_2, highci_2, avg_low_2, avg_high_2 = eval_metric_diff(full_out_data=full_data,
                                                                                                     out_data=fs_data,
                                                                                                     set_no=0.5,
                                                                                                     crnt_k=args.n_features,
                                                                                                     fs_results=fs_results,
                                                                                                     metric="avg_prec")
            fs_results, pval_diff_3, diff_est_3, lowci_3, highci_3, avg_low_3, avg_high_3 = eval_metric_diff(full_out_data=full_data,
                                                                                                     out_data=fs_data,
                                                                                                     set_no=0.5,
                                                                                                     crnt_k=args.n_features,
                                                                                                     fs_results=fs_results,
                                                                                                     metric="logloss")
            print(f"Selected top {args.n_features} features model as final FS model. {args.n_features} \
                            \nfeatures AUC p-value difference to full model: {pval_diff:.4f}, \
                            \nAUC diff estimate: {diff_est:.4f} (CI: {lowci:.4f}-{highci:.4f}), \
                            \nFull model AUC: {avg_low_1:.4f}, FS model AUC: {avg_high_1:.4f} \
                            \n\nAvgPrec p-value difference to full model: {pval_diff_2:.4f}, \
                            \n AvgPrec diff estimate: {diff_est_2:.4f} (CI: {lowci_2:.4f}-{highci_2:.4f}), \
                            \nFull model AvgPrec: {avg_low_2:.4f}, FS model AvgPrec: {avg_high_2:.4f} \
                            \n\nLogloss p-value difference to full model: {pval_diff_3:.4f}, \
                            \n Logloss diff estimate: {diff_est_3:.4f} (CI: {lowci_3:.4f}-{highci_3:.4f}), \
                            \nFull model Logloss: {avg_low_3:.4f}, FS model Logloss: {avg_high_3:.4f} \
                            ")
            fs_results.write_csv(out_model_dir + "fs_auc_diffs_" + get_date() + ".tsv", separator="\t")
            args.n_features = args.n_features
            X_fs_all.with_columns(pl.Series("FINNGENID", fgids)).write_parquet(out_model_dir + "Xall_unscaled_" + "fs_"+ str(args.n_features) + "_" + get_date() + ".parquet")


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving or loading                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #           
    model_fs_cv = xgb_final_fitting(best_params=best_fs_params,
                                    X_train=X_fs_train, y_train=y_train, 
                                    X_finetune_valid=X_fs_finetune_valid, y_finetune_valid=y_finetune_valid,
                                    X_valid=X_fs_valid, y_valid=y_valid, 
                                    X_test=X_fs_test, y_test=y_test, 
                                    metric=args.metric,
                                    low_lr=args.low_lr,
                                    early_stop=args.early_stop,
                                    n_classes=len(y_train.unique()),
                                    fit_cv=True,
                                    final_fit=args.final_fit)
    shap_explainer = shap.TreeExplainer(model_fs_cv)
    train_importances, _ = get_shap_importances(X_fs_train, 
                                                shap_explainer, 
                                                args.lab_name, 
                                                args.lab_name_two, 
                                                translate=True)
    save_importances(top_gain=train_importances,
                                    out_down_path=out_down_path,
                                    study_name=study_name,
                                    lab_name=args.lab_name,
                                    goal=args.goal,
                                    subset="train",
                                    n_features=args.n_features)
    model_fs_cv.save_model(out_model_dir + "model_fs" + str(args.n_features) + "_cv_" + get_date() + ".pkl")  
    pickle.dump(scaler_base, open(out_model_dir + "scaler_" + get_date() + ".pkl", "wb"))

    fs_cv_data = get_out_data(data=data, 
                                    model_fs=model_fs_cv, 
                                    X_all=X_fs_all, 
                                    y_all=y_all, 
                                    metric=args.metric,
                                    lab_name=args.lab_name,
                                    goal=args.goal,
                                    abnorm_extra_choice=args.abnorm_extra_choice)
    fs_cv_data.write_csv(out_model_dir + "preds_fs" + str(args.n_features) + "_cv_" + get_date() + ".tsv", separator="\t")  
    fs_cv_data.write_parquet(out_model_dir + "preds_fs" + str(args.n_features) + "_cv_" + get_date() + ".parquet")  

    save_all_report_plots(out_data=fs_cv_data,
                          out_plot_path=out_plot_path,
                          out_down_path=out_down_path,
                          study_name=study_name,
                          train_importances=train_importances,
                          valid_importances=train_importances,
                          test_importances=train_importances,
                          train_type=get_train_type(args.metric),
                          model_type=args.model_type,
                          fit_cv=args.fit_cv,
                          n_features=args.n_features)

