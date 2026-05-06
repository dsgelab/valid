# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer, get_common_schema, align_schema

from model_eval_utils import get_train_type, save_all_report_plots
from optuna_utils import run_optuna_optim_cv
from xgb_utils import create_xgb_dts, get_shap_importances, save_importances, get_out_data
from input_utils import get_data_and_pred_list
from model_fit_utils import xgb_final_fitting, get_xgb_base_params
from labeling_utils import log_print_n

# Standard stuff
import polars as pl
import shap

# Logging and input
import argparse
import logging
logger = logging.getLogger(__name__)

# Output
import xgboost as xgb

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
    parser.add_argument("--clean", type=int, default=0, help="Whether to remove lab predictors that are very similar or just the lab itself to the predicted lab value to avoid data leakage. [default: 0 = do not clean]")
    parser.add_argument("--file_path_sumstats", type=str, help="Path to summary statistics of a single lab value data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_second_sumstats", type=str, help="Path to summary statistics of a another lab value data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_pgs1", type=str, help="PGS scores - 1/2", default="")
    parser.add_argument("--file_path_pgs2", type=str, help="PGS scores - 2/2", default="")
    parser.add_argument("--file_path_transformer", type=str, help="Transformer outputs", default="")

    parser.add_argument("--file_path_val_icds", type=str, help="Path to ICD data for the future validation set. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_val_atcs", type=str, help="Path to ATC data for the future validation set. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_val_labs", type=str, help="Path to Lab data for the future validation set. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_val_sumstats", type=str, help="Path to summary statistics of a single lab value data for the future validation set. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_val_second_sumstats", type=str, help="Path to summary statistics of a another lab value data for the future validation set. Each column is a predictor. [default: '' = not loaded]", default="")

    parser.add_argument("--final_indvs_path", type=str, help="Path to DF with IDs of final individuals.", default="")

    # Extra info
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--lab_name_two", type=str, help="Readable name of the second most relevant measurement value for file naming.", default="")
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before")
    parser.add_argument("--val_start_date", type=str, default="", help="Date to filter before")

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
    parser.add_argument("--device", type=str, help="Device to use for XGBoost training.", default="cpu")
    parser.add_argument("--nthread", type=int, help="Threads for trainig", default=1)

    # Hyperparameter optimization parameters
    parser.add_argument("--n_trials", type=int, help="Number of hyperparameter optimizations to run [default: 1 = running based on time_step1 instead]", default=1)
    parser.add_argument("--time_optim", type=int, help="Number of seconds to run hyperparameter optimizations for, instead of basing it on the number of traisl. [run when n_trials=1]", default=300)
    parser.add_argument("--refit", type=int, help="Whether to rerun the hyperparameter optimization", default=1)


    # Final model fitting and evaluation
    parser.add_argument("--skip_model_fit", type=int, help="Whether to rerun the final model fitting, or load a prior model fit.", default=0)
    parser.add_argument("--skip_shaps", type=int, help="Whether to rerun shap calculations.", default=0)

    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics.", default=500)
    parser.add_argument("--final_fit", type=int, help="Do one final fit on all data.", default=0)
    parser.add_argument("--fids_path", type=str, help="IDs to filter data for.", default="")
    parser.add_argument("--filter_name", type=str, help="Name of ID filter.", default="")

    parser.add_argument("--train_pct", type=int, help="Percentage of training data to use. Note that this it the percentage of training and not fine-tuning data.", default=100)
    parser.add_argument("--future_val", type=int, default=0, help="Percentage of future validation data to use. Note that this is the percentage of future validation and not training data.")


    args = parser.parse_args()

    return(args)


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
    if not args.future_val:
        data, X_cols = get_data_and_pred_list(file_path_labels=args.file_path_labels, 
                                              file_path_icds=args.file_path_icds, 
                                              file_path_atcs=args.file_path_atcs, 
                                              file_path_sumstats=args.file_path_sumstats, 
                                              file_path_second_sumstats=args.file_path_second_sumstats, 
                                              file_path_labs=args.file_path_labs, 
                                              lab_name=args.lab_name,
                                              clean=args.clean,
                                              file_path_pgs1=args.file_path_pgs1,
                                              file_path_pgs2=args.file_path_pgs2,
                                              file_path_transformer=args.file_path_transformer,
                                              preds=args.preds,
                                              start_date=args.start_date,
                                              fill_missing=0 if args.model_type=="xgb" else 1,
                                              fids_path=args.fids_path,
                                              fg_ver=args.fg_ver)
        fgids, X_train, y_train, X_finetune_valid, y_finetune_valid, \
            X_valid, y_valid, X_test, y_test, X_all, y_all, \
            X_val_all, y_val_all, \
            dtrain, dfinetunevalid, dvalid, \
                data, val_data = create_xgb_dts(data=data, 
                                                X_cols=X_cols, 
                                                y_goal=args.goal,
                                                train_pct=args.train_pct)
        log_print_n(data.filter(pl.col.SET==0.5), "Finetune Valid")
    else:
        data, X_cols = get_data_and_pred_list(file_path_labels=args.file_path_labels, 
                                              file_path_icds=args.file_path_icds, 
                                              file_path_atcs=args.file_path_atcs, 
                                              file_path_sumstats=args.file_path_sumstats, 
                                              file_path_second_sumstats=args.file_path_second_sumstats, 
                                              file_path_labs=args.file_path_labs,
                                              lab_name=args.lab_name,
                                              clean=args.clean, 
                                              file_path_pgs1=args.file_path_pgs1,
                                              file_path_pgs2=args.file_path_pgs2,
                                              file_path_transformer=args.file_path_transformer,
                                              preds=args.preds,
                                              start_date=args.start_date,
                                              fill_missing=0 if args.model_type=="xgb" else 1,
                                              fids_path=args.fids_path,
                                              future_val="train" if not args.final_fit else "final train",
                                              fg_ver=args.fg_ver)
        val_data, val_X_cols = get_data_and_pred_list(file_path_labels=args.file_path_labels if not args.final_fit else args.final_indvs_path, 
                                                      file_path_icds=args.file_path_val_icds, 
                                                      file_path_atcs=args.file_path_val_atcs, 
                                                      file_path_sumstats=args.file_path_val_sumstats, 
                                                      file_path_second_sumstats=args.file_path_val_second_sumstats, 
                                                      file_path_labs=args.file_path_val_labs, 
                                                      lab_name=args.lab_name,
                                                      clean=args.clean,
                                                      file_path_pgs1=args.file_path_pgs1,
                                                      file_path_pgs2=args.file_path_pgs2,
                                                      file_path_transformer=args.file_path_transformer,
                                                      preds=args.preds,
                                                      start_date=args.val_start_date,
                                                      fill_missing=0 if args.model_type=="xgb" else 1,
                                                      fids_path=args.fids_path,
                                                      future_val="val" if not args.final_fit else "final val",
                                                      fg_ver=args.fg_ver)
        
        fgids, X_train, y_train, X_finetune_valid, y_finetune_valid, \
            X_valid, y_valid, X_test, y_test, X_all, y_all, \
            X_val_all, y_val_all, \
            dtrain, dfinetunevalid, dvalid, \
                data, val_data = create_xgb_dts(data=data, 
                                                X_cols=X_cols, 
                                                y_goal=args.goal,
                                                train_pct=args.train_pct,
                                                val_data=val_data,
                                                final_fit=args.final_fit)
        
    print(X_all.with_columns(pl.Series("FINNGENID", fgids)).head(2))
    log_print_n(data.filter(pl.col.SET==0), "Train")
    log_print_n(val_data.filter(pl.col.SET==1), "Valid")
    log_print_n(val_data.filter(pl.col.SET==2), "Test")
    print(data["SET"].value_counts(normalize=True))
    X_all.with_columns(pl.Series("FINNGENID", fgids)).write_parquet(out_model_dir + "Xall_" + get_date() + ".parquet")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Hyperparam optimization with optuna                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.model_type=="xgb":
        if not args.skip_model_fit:
            base_params = get_xgb_base_params(metric=args.metric, 
                                              lr=args.lr, 
                                              n_classes=len(y_train.unique()),
                                              device=args.device,
                                              nthread=args.nthread)
            best_params = run_optuna_optim_cv(train=[pl.concat([X_train, X_finetune_valid]), pl.concat([y_train, y_finetune_valid])], 
                                              lab_name=args.lab_name, 
                                              refit=args.refit, 
                                              time_optim=args.time_optim, 
                                              n_trials=args.n_trials, 
                                              study_name=study_name,
                                              res_dir=args.res_dir,
                                              model_type="xgb",
                                              model_fit_date=args.model_fit_date,
                                              base_params=base_params,
                                              fg_ver=args.fg_ver)
            logging.info(timer.get_elapsed())
            model_final = xgb_final_fitting(best_params=best_params,
                                            X_train=X_train, y_train=y_train, 
                                            X_finetune_valid=X_finetune_valid,
                                                y_finetune_valid=y_finetune_valid,
                                            metric=args.metric,
                                            low_lr=args.low_lr,
                                            early_stop=args.early_stop,
                                            n_classes=len(y_train.unique()),
                                            device=args.device)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving or loading                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            model_final.save_model(out_model_dir + "model_" + get_date() + ".json")
        else:
            model_final = xgb.XGBClassifier()
            model_final.load_model(out_model_dir + "model_" + args.model_fit_date + ".json")

    if not args.skip_shaps:
        if args.model_type == "xgb" or args.model_type == "cat":
            # SHAP explainer for model
            shap_explainer = shap.TreeExplainer(model_final)
            train_importances, _ = get_shap_importances(X_in=pl.concat([X_train, X_finetune_valid]), 
                                                        explainer=shap_explainer, 
                                                        lab_name=args.lab_name, 
                                                        lab_name_two=args.lab_name_two,
                                                        translate=True,
                                                        device=args.device)
            save_importances(top_gain=train_importances,
                            out_down_path=out_down_path,
                            study_name=study_name,
                            lab_name=args.lab_name,
                            goal=args.goal,
                            subset="train")
            
            valid_importances, _ = get_shap_importances(X_in=X_valid, 
                                                        explainer=shap_explainer, 
                                                        lab_name=args.lab_name, 
                                                        lab_name_two=args.lab_name_two,
                                                        translate=True,
                                                        device=args.device)
            save_importances(top_gain=valid_importances,
                             out_down_path=out_down_path,
                             study_name=study_name,
                             lab_name=args.lab_name,
                             goal=args.goal,
                             subset="valid")
            if X_test.height > 0:
                test_importances, _ = get_shap_importances(X_in=X_test, 
                                                           explainer=shap_explainer, 
                                                           lab_name=args.lab_name, 
                                                           lab_name_two=args.lab_name_two,
                                                           translate=True,
                                                           device=args.device)
                save_importances(top_gain=test_importances,
                                 out_down_path=out_down_path,
                                 study_name=study_name,
                                 lab_name=args.lab_name,
                                 goal=args.goal,
                                 subset="test")
            else:
                test_importances = None    

    else:
        train_importances = pl.read_csv(out_down_path+"/"+args.goal+"/" + args.lab_name+"_"+study_name+"_"+args.goal+"_shap_importance_train_" + args.model_fit_date + ".csv")
        valid_importances = pl.read_csv(out_down_path+"/"+args.goal+"/" + args.lab_name+"_"+study_name+"_"+args.goal+"_shap_importance_valid_" + args.model_fit_date + ".csv")
        if not args.final_fit:
            test_importances = pl.read_csv(out_down_path+"/"+args.goal+"/" + args.lab_name+"_"+study_name+"_"+args.goal+"_shap_importance_test_" + args.model_fit_date + ".csv")
        else:
            test_importances = None


    # Model predictions
    out_data, optimal_proba_cutoff = get_out_data(data=data, 
                            model_final=model_final, 
                            X_all=X_all, 
                            y_all=y_all, 
                            metric=args.metric,
                            lab_name=args.lab_name,
                            goal=args.goal,
                            abnorm_extra_choice=args.abnorm_extra_choice,
                            optimal_proba_cutoff=None,
                            device=args.device)
    val_out_data, _ = get_out_data(data=val_data, 
                                model_final=model_final, 
                                X_all=X_val_all, 
                                y_all=y_val_all, 
                                metric=args.metric,
                                lab_name=args.lab_name,
                                goal=args.goal,
                                abnorm_extra_choice=args.abnorm_extra_choice,
                                optimal_proba_cutoff=optimal_proba_cutoff,
                                device=args.device)
    if not args.final_fit:
        schema = get_common_schema([out_data, val_out_data])
        out_data = pl.concat([align_schema(df, schema) for df in [out_data, val_out_data]])
        out_data = out_data.with_columns(pl.when(pl.col.SET==0.5).then(pl.lit(0)).otherwise(pl.col.SET).alias("SET_fixed"))

        out_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")  
    else:
        out_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")
        val_out_data.write_parquet(out_model_dir + "final_preds_" + get_date() + ".parquet")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # #                 Plotting                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    save_all_report_plots(out_data=out_data,
                          out_plot_path=out_plot_path,
                          out_down_path=out_down_path,
                          study_name=study_name,
                          train_importances=train_importances,
                          valid_importances=valid_importances,
                          test_importances=test_importances,
                          train_type=get_train_type(args.metric),
                          model_type=args.model_type)