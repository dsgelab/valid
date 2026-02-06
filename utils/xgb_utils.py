
import logging
import polars as pl
import xgboost as xgb
import logging
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from model_eval_utils import get_train_type, get_optim_precision_recall_cutoff
from abnorm_utils import get_abnorm_func_based_on_name
from model_fit_utils import get_cont_goal_col_name
def get_out_data(data: pl.DataFrame, 
                 model_final: xgb.XGBClassifier, 
                 X_all: pl.DataFrame, 
                 y_all: pl.DataFrame, 
                 metric: str,
                 lab_name: str,
                 goal: str,
                 abnorm_extra_choice: str="") -> pl.DataFrame:
    """Creates the output data with the predictions of the model.
    Args:
        data (pl.DataFrame): The input data.
        model_final (xgb.XGBClassifier): The trained XGBoost model.
        X_all (pl.DataFrame): The input features for all data.
        y_all (pl.DataFrame): The true labels for all data.
        metric (str): The evaluation metric.
        lab_name (str): The name of the lab.
        goal (str): The goal column name.
        abnorm_extra_choice (str): Additional abnormality choice (default is empty string). 

       Returns the relevant columns from the input data with the predictions of the model.
       Not that abnormality here is the case/control status.
       Columns:
            - `FINNGENID`: Individual ID
            - `EVENT_AGE`: Age at event (for controls last lab record and for cases )
            - `LAST_VAL_DATE`: Prediction date (TODO need to check what exact time this is)
            - `SET`: Train, validation or test set
            - `N_PRIOR_ABNORMS`: Number of prior abnormal values
            - `VALUE` (if continuous prediction): Predicted value
            - `ABNORM_PROBS` (if binary prediction): Probability of abnormality
            - `ABNORM_PREDS` (if binary prediction): Predicted abnormality
            - `TRUE_VALUE` (if continuous prediction): True value
            - `TRUE_ABNORM` (if binary prediction): True abnormality"""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Base data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # data = data.with_columns([
    #     pl.when(pl.col(col)>0).then(1).otherwise(0).alias(col) for col in data.columns if "ABNORM" in col
    # ])
    out_data = (data
                .select("FINNGENID", "SEX", "y_MEAN", "y_MEAN_ABNORM", "y_MIN", "y_MIN_ABNORM", "y_NEXT", "y_NEXT_ABNORM", "EVENT_AGE", "SET", "LAST_VAL_DATE", "ABNORM")
                .rename({"LAST_VAL_DATE": "DATE", "ABNORM": "N_PRIOR_ABNORMS"})
               )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Continuous prediction                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ################# PROBABLY DEPRECATED #####################################
    if get_train_type(metric) == "cont":    
        y_pred = model_final.predict(X_all.to_numpy())
        out_data = out_data.with_columns([
                            pl.Series("ABNORM_PROBS", y_pred),
                            pl.Series("TRUE_VALUE", y_all),
        ])
        # Abnormality prediction based on the continuous value
        out_data = get_abnorm_func_based_on_name(lab_name, abnorm_extra_choice)(out_data, "ABNORM_PROBS").rename({"ABNORM_CUSTOM": "ABNORM_PREDS"})
        out_data = get_abnorm_func_based_on_name(lab_name, abnorm_extra_choice)(out_data, "TRUE_VALUE").rename({"ABNORM_CUSTOM": "TRUE_ABNORM"})
        print(out_data["TRUE_ABNORM"].value_counts())
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Binary prediction of "abnormality"                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    else:        
        # # # # # # # # # Continuous column # # # # # # # # # # # # # # # # # # # #
        new_goal = get_cont_goal_col_name(goal, data.columns)
        if new_goal is not None:
            y_cont_all = data.select(new_goal)

            out_data = out_data.with_columns([
                pl.Series("TRUE_VALUE", y_cont_all),
            ])
        out_data = out_data.with_columns([pl.col(goal).alias("TRUE_ABNORM")])

        # # # # # # # # # Abnormality columns # # # # # # # # # # # # # # # # # # #
        if get_train_type(metric) == "bin":  
            y_pred = model_final.predict_proba(X_all.to_numpy())[:,1]
            out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS")])
            # Binary abnormality prediction based on optimal cut-off
            optimal_proba_cutoff = get_optim_precision_recall_cutoff(out_data)
            logging.info(f"Optimal cut-off for prediction based on PR {optimal_proba_cutoff}")
            out_data = out_data.with_columns(
                (pl.col("ABNORM_PROBS") > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS")
            )
        elif get_train_type(metric) == "multi":  
            for pred_val in out_data["TRUE_ABNORM"].unique():
                y_pred = model_final.predict_proba(X_all.to_numpy())[:,pred_val]
                out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS_"+str(pred_val))])
                # Binary abnormality prediction based on optimal cut-off
                temp_data = out_data.with_columns(pl.when((pl.col.TRUE_ABNORM!=1)&(pl.col.TRUE_ABNORM!=0)).then(pl.lit(0)).otherwise(pl.col.TRUE_ABNORM).alias("TRUE_ABNORM"))
                temp_data = temp_data.rename({"ABNORM_PROBS_"+str(pred_val): "ABNORM_PROBS"})
                optimal_proba_cutoff = get_optim_precision_recall_cutoff(temp_data)
                logging.info(f"Optimal cut-off for {pred_val} prediction based on PR {optimal_proba_cutoff}")
                out_data = out_data.with_columns(
                (pl.col("ABNORM_PROBS_"+str(pred_val)) > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS_"+str(pred_val)))
    return(out_data)

import polars as pl
import shap
import numpy as np
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from minor_plot_utils import get_plot_names
def get_shap_importances(X_in: pl.DataFrame,
                         explainer: shap.TreeExplainer,
                         lab_name: str,
                         lab_name_two: str="",
                         translate: bool=True) -> tuple[pl.DataFrame, list]:
    """Returns the SHAP importances of the model.
    Args:
        X_in (pl.DataFrame): The input data.
        explainer (shap.TreeExplainer): The SHAP explainer for the model.
        lab_name (str): The name of the lab.
        lab_name_two (str): The name of the second lab (default is empty string).
    Returns:
        tuple: A tuple containing the SHAP importances (pl.DataFrame) and the feature names (list)."""
    if translate:
        new_names = get_plot_names(X_in.columns, lab_name, lab_name_two)
    else:
        new_names = X_in.columns
    explanations = explainer.shap_values(X_in.to_pandas())
    mean_shaps = np.abs(explanations).mean(0)
    shap_importance = pl.DataFrame({"mean_shap": mean_shaps}, schema=["mean_shap"]).with_columns(pl.Series("labels", new_names)).sort("mean_shap", descending=True)

    return(shap_importance, new_names)

import polars as pl
import logging
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import make_dir, get_date
def save_importances(top_gain,
                     out_down_path: str,
                     study_name: str,
                     goal,
                     lab_name,
                     subset: str="all",
                     n_features: int=None,
                     cv: bool=False) -> None:
    """Saves the feature importances of the model to a csv file.
    Args:
        top_gain (pl.DataFrame): The dataframe containing the feature importances.
        out_down_path (str): The output directory path.
        subset (str): The subset of the data (default is "all").
    Returns:
        None"""
    crnt_out_down_path = out_down_path+"/"+goal+"/"; make_dir(crnt_out_down_path)
    if n_features is None:
        if cv:
            file_path = crnt_out_down_path + lab_name+"_"+study_name+"_"+goal+"_shap_importance_cv_" + subset + "_" + get_date() + ".csv"
        else:
            file_path = crnt_out_down_path + lab_name+"_"+study_name+"_"+goal+"_shap_importance_" + subset + "_" + get_date() + ".csv"
    else:
        file_path = crnt_out_down_path + lab_name+"_"+study_name+"_"+goal+"_shap_importance_fs"+str(n_features)+"_" + subset + "_" + get_date() + ".csv"
    try:
        logging.info(top_gain.head(10))
        top_gain.write_csv(file_path)
    except:
        top_gain = pl.DataFrame(top_gain).with_columns(pl.col(top_gain.columns[0]).arr.to_struct().alias(top_gain.columns[0])).unnest(top_gain.columns[0])
        logging.info(top_gain.head(10))
        top_gain.write_csv(file_path)


import polars as pl
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple
def create_xgb_dts(data: pl.DataFrame, 
                   X_cols: list, 
                   y_goal: str,
                   train_pct: int = 1) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, xgb.DMatrix, xgb.DMatrix, StandardScaler]:
    """Creates the XGBoost data matrices and scales the data.

    Returns the data and the predictors.
    
        Args:
            data (pl.DataFrame): The input data.
            X_cols (list): The list of feature columns.
            y_goal (str): The target column.
            reweight (int): The reweighting factor.

        Returns:
            tuple: A tuple containing the data (polars DataFrame) and the list of predictors
    """
    # Need later to be Float for scaling
    data = data.with_columns([
        pl.col(col).cast(pl.Float64, strict=False) for col, dtype in data.schema.items() if isinstance(dtype, pl.Int64) or isinstance(dtype, pl.Int32)
    ])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Replace -1s with 2s                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if -1 in data.get_column(y_goal):
        data = data.with_columns(pl.when(pl.col(y_goal)==-1).then(pl.lit(2)).otherwise(pl.col(y_goal)).alias(y_goal))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Scaling                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    binary_cols = ([col_name for col_name in X_cols
                       if(set(data.select(pl.col(col_name).drop_nulls().unique()).to_series()) <= {0, 1} or
                          set(data.select(pl.col(col_name).drop_nulls().unique()).to_series()) <= {"0", "1"})  
    ])
    print(binary_cols)
    numeric_cols = [col_name for col_name in X_cols if col_name not in binary_cols]
    print(numeric_cols)
    scaler_base = StandardScaler()
    scaler_base.set_output(transform="polars")
    X_cols = numeric_cols + binary_cols
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Split data                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    train_data = data.filter(pl.col("SET")==0).drop("SET")

    finetune_valid_data = data.filter(pl.col("SET")==0.5).drop("SET")

    # if training pct<100, take only part of training data
    if train_pct<100:
        train_data = train_data.sample(fraction=train_pct/100, with_replacement=False, seed=42)
        data = data.filter(pl.col("FINNGENID").is_in(train_data["FINNGENID"]).or_(pl.col("SET")!=0))
        finetune_valid_data = finetune_valid_data.sample(fraction=train_pct/100, with_replacement=False, seed=42)
        data = data.filter(pl.col("FINNGENID").is_in(finetune_valid_data["FINNGENID"]).or_(pl.col("SET")!=0.5))

    valid_data = data.filter(pl.col("SET")==1).drop("SET")
    test_data = data.filter(pl.col("SET")==2).drop("SET")
        
    X_train = train_data.select(X_cols); y_train = train_data.select(y_goal)
    if finetune_valid_data.height>0:
        X_finetune_valid = finetune_valid_data.select(X_cols); y_finetune_valid = finetune_valid_data.select(y_goal)
    else:
        X_finetune_valid = pl.DataFrame({col: [] for col in X_cols}); y_finetune_valid = pl.DataFrame({y_goal: []})
    X_valid = valid_data.select(X_cols); y_valid = valid_data.select(y_goal)
    X_test = test_data.select(X_cols); y_test = test_data.select(y_goal)
    X_all = data.select(X_cols); y_all = data.select(y_goal)

    preprocessor = ColumnTransformer([
        ('num', scaler_base, numeric_cols),
        ('bin', 'passthrough', binary_cols)
    ])
    X_train=pl.DataFrame(preprocessor.fit_transform(X_train), schema=X_train.schema)
    if finetune_valid_data.height>0:
        X_finetune_valid=pl.DataFrame(preprocessor.transform(X_finetune_valid), schema=X_finetune_valid.schema)
    X_valid=pl.DataFrame(preprocessor.transform(X_valid), schema=X_valid.schema)
    X_test=pl.DataFrame(preprocessor.transform(X_test), schema=X_test.schema)
    X_all_scaled=pl.DataFrame(preprocessor.transform(X_all), schema=X_all.schema)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 XGBoost datatype                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
    if finetune_valid_data.height>0:
        dfinetunevalid = xgb.DMatrix(data=X_finetune_valid, label=y_finetune_valid, enable_categorical=True)
    else:
        dfinetunevalid = None
    return(data["FINNGENID"], X_train, y_train, X_finetune_valid, y_finetune_valid, X_valid, y_valid, X_test, y_test, X_all_scaled, y_all, X_all, dtrain, dfinetunevalid, dvalid, preprocessor, data)

