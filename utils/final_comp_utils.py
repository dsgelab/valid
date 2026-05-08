import sys
sys.path.append("../../utils/")
from general_utils import get_dated_path
from model_eval_utils import bootstrap_metric, bootstrap_difference, continuous_nri, bootstrap_nri
import sklearn.metrics as skm
import numpy as np
from delong_utils import delong_roc_test
import polars as pl
def final_diff_eval(lab_name,
                    no_dups_combos,
                    base_path,
                    file_descr,
                    set_names,
                    filters,
                    n_boots=100,
                   train_goals=[ "y_MEAN_ABNORM"]):
    results = pl.DataFrame()
    for combo_1, combo_2 in no_dups_combos:
        for train_goal_1 in train_goals:
            for train_goal_2 in train_goals:
                # Getting file paths and metrics
                file_path_1 = get_dated_path(base_path+"/step5_predict/"+file_descr+"/"+train_goal_1+"/"+combo_1+"/models/"+lab_name+"/")
                metric_1 = combo_1.split("_")[1]
    
                file_path_2 = get_dated_path(base_path+"/step5_predict/"+file_descr+"/"+train_goal_2+"/"+combo_2+"/models/"+lab_name+"/")
                metric_2 = combo_2.split("_")[1]
                
                if not file_path_1 or not file_path_2 or file_path_1==file_path_2: continue

                # Getting data
                preds_1 = pl.read_parquet(file_path_1)
                if "ABNORM_PROBS_1" in preds_1.columns: preds_1 = (preds_1.rename({"ABNORM_PROBS_1": "ABNORM_PROBS"}))
                preds_2 = pl.read_parquet(file_path_2)
                if "ABNORM_PROBS_1" in preds_2.columns: preds_2 = preds_2.rename({"ABNORM_PROBS_1": "ABNORM_PROBS"})
                
                if ((metric_1 == "logloss") or (metric_1 == "mlogloss")) and ((metric_2 == "logloss") or (metric_2 == "mlogloss")):
                    
                    for set_no in set_names:
                        descriptors = {"MODEL_1": combo_1, "TRAIN_1": train_goal_1, "MODEL_2": combo_2, "TRAIN_2": train_goal_2, "SET": set_names[set_no]}
                        # Filtering
                        for crnt_filter_name in filters:
                            descriptors["FILTER"] = crnt_filter_name
            
                            for goal_name in ["y_MEAN_ABNORM"]:
                                
                                descriptors["EVAL"]=goal_name

                                # Aligning predictions
                                preds = (preds_1.select("FINNGENID", "SET", "EVENT_AGE", goal_name, "ABNORM_PROBS")
                                                .with_columns(pl.when(pl.col(goal_name)==0).then(pl.lit(0)).otherwise(pl.lit(1)).alias(goal_name))
                                                .join(preds_2.select("FINNGENID", "ABNORM_PROBS"), on="FINNGENID", how="left")
                                                .filter(~pl.col.ABNORM_PROBS.is_null(), ~pl.col.ABNORM_PROBS_right.is_null())
                                        )
                                if set_no==10:
                                    # this means set is training+validation
                                    crnt_preds = preds.filter(pl.col.SET!=2).filter(filters[crnt_filter_name]) 
                                else:
                                    crnt_preds = preds.filter(pl.col.SET==set_no).filter(filters[crnt_filter_name])

                                # Skipping low counts
                                if (crnt_preds[goal_name]==1).sum()<10: 
                                    print("skipping: "+crnt_filter_name)
                                    continue

                                # Delong p-value of difference
                                pval_diff = 10**delong_roc_test(crnt_preds[goal_name].to_numpy(), crnt_preds["ABNORM_PROBS"].to_numpy(), crnt_preds["ABNORM_PROBS_right"].to_numpy())[0]
    
                                ### Bootstrapping AUC interval
                                diff_est, lowci, highci, _, avg_1, avg_2 = bootstrap_difference(metric_func = (skm.roc_auc_score),
                                                                                                        preds_1=crnt_preds["ABNORM_PROBS"].to_numpy(), 
                                                                                                        preds_2=crnt_preds["ABNORM_PROBS_right"].to_numpy(),
                                                                                                        obs=crnt_preds[goal_name].to_numpy(),
                                                                                                        n_boots=n_boots)
                                descriptors["AUCDiff_Pvalue"]=pval_diff
                                descriptors["AUCDiff_Est"]=diff_est
                                descriptors["AUCDiff_CIneg"]=lowci
                                descriptors["AUCDiff_CIpos"]=highci
                                descriptors["AUCDiff_AUC1"]=avg_1
                                descriptors["AUCDiff_AUC2"]=avg_2
    
                                ### P-values for Average Precision with Bootstrapping
                                diff_est, lowci, highci, pval_diff, avg_1, avg_2 = bootstrap_difference(metric_func = (skm.average_precision_score),
                                                                                                        preds_1=crnt_preds["ABNORM_PROBS"].to_numpy(), 
                                                                                                        preds_2=crnt_preds["ABNORM_PROBS_right"].to_numpy(),
                                                                                                        obs=crnt_preds[goal_name].to_numpy(),
                                                                                                        n_boots=n_boots)
                                descriptors["AvgPrecDiff_Pvalue"]=pval_diff
                                descriptors["AvgPrecDiff_Est"]=diff_est
                                descriptors["AvgPrecDiff_CIneg"]=lowci
                                descriptors["AvgPrecDiff_CIpos"]=highci
                                descriptors["AvgPrecDiff_AvgPrec1"]=avg_1
                                descriptors["AvgPrecDiff_AvgPrec2"]=avg_2
    
            
                                ### NRI with CI measure of if new model is better at reclassification <0 -> worst and >0 -> better
                                nri, lowci, highci = bootstrap_nri(continuous_nri, 
                                                                   crnt_preds[goal_name].to_numpy(), 
                                                                   crnt_preds["ABNORM_PROBS"].to_numpy(),
                                                                   crnt_preds["ABNORM_PROBS_right"].to_numpy(),
                                                                   n_boots=n_boots)
                                descriptors["NRI"]=nri
                                descriptors["NRI_CI"]="("+str(round(lowci, 2))+ "-"+ str(round(highci, 2)) + ")"
            
                                results = pl.concat([results, pl.DataFrame(descriptors)])
                                
                display(results.filter(pl.col.FILTER=="History", pl.col.SET.is_in(["Test", "Valid"]), pl.col.EVAL=="y_MEAN_ABNORM"))
                display(results.filter(pl.col.FILTER=="No history", pl.col.SET.is_in(["Test", "Valid"]), pl.col.EVAL=="y_MEAN_ABNORM"))
    return(results)

from general_utils import get_dated_path
from model_eval_utils import bootstrap_metric
import sklearn.metrics as skm
import numpy as np
import polars as pl
def final_aucs(lab_name,
               base_path,
               labels_path,
               no_dups_combos,
               file_descr,
               goal_names,
               filters,
               set_names,
               n_boots=100,
               train_goals=["y_MEAN_ABNORM"],
               eval_goals=["y_MEAN_ABNORM"],
               eval_sets=[0,1,2]):
    
    results = pl.DataFrame()
    labels = pl.read_parquet(labels_path)
    for crnt_option in set([elem_1 for elem_1, elem_2 in no_dups_combos] + [elem_2 for elem_1, elem_2 in no_dups_combos]):
        for train_goal in train_goals:
            # Getting file paths and metrics
            metric = crnt_option.split("_")[1]
            if metric != "logloss" and metric != "mlogloss": continue
            mdl_name = crnt_option.split("_")[0]
            crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["lr", "xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
            
            # Getting data
            file_path = get_dated_path(base_path+"/step5_predict/"+file_descr+"/"+train_goal+"/"+crnt_option+"/models/"+lab_name+"/")
            if not file_path: continue
            date = file_path.split(".")[0].split("preds_")[1]
            preds = pl.read_parquet(file_path)
            if "ABNORM_PROBS_1" in preds.columns: preds = (preds.rename({"ABNORM_PROBS_1": "ABNORM_PROBS"}))


            # setup results
            crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["lr", "xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
            descriptors = {"Date": date, 
                           "Model": mdl_name, 
                           "Train Goal": train_goal, 
                           "File Description": file_descr, 
                           "Predictors": crnt_pred_descr,
                           "Outcome": goal_names[train_goal]}
            
            for crnt_filter_name in filters:
                filter_preds = preds.filter(filters[crnt_filter_name])
                
                for set_no in eval_sets:
                    if set_no == 10: crnt_preds=filter_preds.filter(pl.col.SET!=2)
                    else: crnt_preds=filter_preds.filter(pl.col.SET==set_no)
                    if crnt_preds.height == 0: continue
                        
                    descriptors["FILTER"] = crnt_filter_name
                    descriptors["SET"] = set_names[set_no]
                    for goal in eval_goals:
                        crnt_preds = (crnt_preds.with_columns(pl.when(pl.col(goal)==0).then(pl.lit(0)).otherwise(pl.lit(1)).alias(goal)))
                        descriptors["Eval goal"] = goal
                        N_total = crnt_preds.height
                        N_cases = crnt_preds.filter(pl.col(goal)==1).height
                        descriptors["N_TOTAL"] = N_total
                        descriptors["N_CASE"] = N_cases

                        ### AUC
                        AUC = bootstrap_metric(skm.roc_auc_score, 
                                               crnt_preds[goal],
                                               crnt_preds["ABNORM_PROBS"],
                                               n_boots=n_boots)
                        descriptors["AUC"] = AUC[0]
                        descriptors["AUC_CIneg"] = AUC[1]
                        descriptors["AUC_CIpos"] = AUC[2]

                        ### Average Precision
                        averagePrec = bootstrap_metric(skm.average_precision_score, 
                                                   crnt_preds[goal],
                                                   crnt_preds["ABNORM_PROBS"],
                                                   n_boots=n_boots)
                        descriptors["avgPrec"] = averagePrec[0]
                        descriptors["avgPrec_CIneg"] = averagePrec[1]
                        descriptors["avgPrec_CIpos"] = averagePrec[2]

                        ### Brier
                        brier = bootstrap_metric(skm.brier_score_loss, 
                                                       crnt_preds[goal],
                                                       crnt_preds["ABNORM_PROBS"],
                                                       n_boots=n_boots)
                        descriptors["Brier"] = brier[0]
                        descriptors["Brier_CIneg"] = brier[1]
                        descriptors["Brier_CIpos"] = brier[2]

                        ### Logloss
                        logloss = bootstrap_metric(skm.log_loss, 
                                                       crnt_preds[goal],
                                                       crnt_preds["ABNORM_PROBS"],
                                                       n_boots=n_boots)
                        descriptors["Logloss"] = logloss[0]
                        descriptors["Logloss_CIneg"] = logloss[1]
                        descriptors["Logloss_CIpos"] = logloss[2]

                    
                        results = pl.concat([results, pl.DataFrame(descriptors)])
            display(results)
    return(results)