# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer
from model_eval_utils import get_train_type, save_all_report_plots
from optuna_utils import run_optuna_optim
from xgb_utils import create_xgb_dts, get_shap_importances, save_importances, get_out_data
from input_utils import get_data_and_pred_list   
from model_fit_utils import xgb_final_fitting, get_xgb_base_params

# Standard stuff
import polars as pl

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

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    timer = Timer()
    args = get_parser_arguments()

    # File names
    study_name = "xgb_" + str(args.metric) + "_" + args.pred_descriptor
    if args.model_fit_date == "": args.model_fit_date = get_date()

    out_dir = args.res_dir + study_name + "/"; 
    out_model_dir = out_dir + "models/" + args.lab_name + "/" + args.model_fit_date + "/" 
    out_plot_dir = out_dir + "plots/" + args.model_fit_date + "/"
    out_plot_path = out_plot_dir + args.lab_name + "_"
    out_down_dir = out_dir + "down/" + args.model_fit_date + "/"
    out_down_path = out_down_dir + args.lab_name + "_" + study_name + "_"
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
                                          preds=args.preds,
                                          start_date=args.start_date)
    fgids, X_train, y_train, X_valid, y_valid, \
        X_test, y_test, X_all, y_all, X_all_unscaled, \
        dtrain, dvalid, scaler_base = create_xgb_dts(data=data, 
                                                     X_cols=X_cols, 
                                                     y_goal=args.goal)
    print(X_all_unscaled.with_columns(pl.Series("FINNGENID", fgids)).head(2))
    X_all_unscaled.with_columns(pl.Series("FINNGENID", fgids)).write_parquet(out_model_dir + "Xall_unscaled_" + get_date() + ".parquet")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Hyperparam optimization with optuna                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if not args.skip_model_fit:
        base_params = get_xgb_base_params(metric=args.metric, 
                                          lr=args.lr, 
                                          n_classes=len(y_train.unique()))
        best_params = run_optuna_optim(train=dtrain, 
                                       valid=dvalid, 
                                       test=None,
                                       lab_name=args.lab_name, 
                                       refit=args.refit, 
                                       time_optim=args.time_optim, 
                                       n_trials=args.n_trials, 
                                       study_name=study_name,
                                       res_dir=args.res_dir,
                                       model_type="xgb",
                                       model_fit_date=args.model_fit_date,
                                       base_params=base_params)
        logging.info(timer.get_elapsed())
        model_final = xgb_final_fitting(best_params=best_params, 
                                        X_train=X_train, y_train=y_train, 
                                        X_valid=X_valid, y_valid=y_valid, 
                                        metric=args.metric,
                                        low_lr=args.low_lr,
                                        early_stop=args.early_stop,
                                        n_classes=len(y_train.unique()))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving or loading                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        pickle.dump(model_final, open(out_model_dir + "model_" + get_date() + ".pkl", "wb"))
        pickle.dump(scaler_base, open(out_model_dir + "scaler_" + get_date() + ".pkl", "wb"))
    else:
        model_final = pickle.load(open(out_model_dir + "model_" + get_date() + ".pkl", "rb"))

    if not args.skip_model_fit:
        # SHAP explainer for model
        shap_explainer = shap.TreeExplainer(model_final)
        train_importances, _ = get_shap_importances(X_train, shap_explainer, args.lab_name, args.lab_name_two)
        test_importances, _ = get_shap_importances(X_test, shap_explainer, args.lab_name, args.lab_name_two)
        save_importances(top_gain=train_importances,
                         out_down_path=out_down_path,
                         subset="train")
        save_importances(top_gain=test_importances,
                         out_down_path=out_down_path,
                         subset="test")
        # Model predictions
        out_data = get_out_data(data=data, 
                                model_final=model_final, 
                                X_all=X_all, 
                                y_all=y_all, 
                                metric=args.metric,
                                lab_name=args.lab_name,
                                goal=args.goal,
                                abnorm_extra_choice=args.abnorm_extra_choice)
        out_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")  
    else:
        train_importances = pl.read_csv(out_down_path + "shap_importance_train_" + args.model_fit_date + ".csv")
        test_importances = pl.read_csv(out_down_path + "shap_importance_test_" + args.model_fit_date + ".csv")
        out_data = pl.read_parquet(out_model_dir + "preds_" + get_date() + ".parquet")


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # #                 Plotting                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
    save_all_report_plots(out_data=out_data,
                          train_importances=train_importances,
                          valid_importances=train_importances,
                          test_importances=test_importances,
                          out_plot_path=out_plot_path,
                          out_down_path=out_down_path,
                          train_type=get_train_type(args.metric))

