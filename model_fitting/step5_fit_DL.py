try:
    from termcolor import colored
except:
    def colored(string, color): return(string)
########## Checked
import sys

sys.path.append(("/home/ivm/valid/scripts/pytorch_ehr/"))
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer
from torch_utils import get_mbs_from_files, get_model, get_torch_optimizer, get_out_data
from optuna_utils import run_optuna_optim
from utils_final import epochs_run
from model_eval_utils import save_all_report_plots, get_all_eval_metrics, create_report

# Logging and input
import argparse
import logging
logger = logging.getLogger(__name__)

# General
import torch
import polars as pl
import pickle
from io import open


def get_parser_arguments():
    #this is where you define all the things you wanna run in your main file
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')
    
    # Data paths
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument('--file_path', type = str, required=True, help='the path to the folders with pickled file(s)')
    parser.add_argument('--file_name_start', type = str, required=True, help='the path to the folders with pickled file(s)')

    # Extra info
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--model_fit_date", type=str, help="Original date of model fitting.", default="")
    # Model fitting best_params
    parser.add_argument('--model_name', type=str, default='RNN',choices= ['RNN','DRNN','QRNN','TLSTM','LR','RETAIN'], help='choose from {"RNN","DRNN","QRNN","TLSTM","LR","RETAIN"}') 
    parser.add_argument('--simple_fit', type=int, default=0, help="Runs a simple fit of the model with no hyperparameter tuning. [default: 0]") 
    parser.add_argument("--early_stop", type=int, help="Early stopping for the final fitting round. Currently, early stopping fixed at 5 for hyperparameter optimization.", default=5)
    # Preset best_params
    parser.add_argument('--cell_type', type = str, default = 'GRU', choices=['RNN', 'GRU', 'LSTM'], help='For RNN based models, choose from {"RNN", "GRU", "LSTM", "QRNN" (for QRNN model only)}, "TLSTM (for TLSTM model only')
    parser.add_argument('-input_size', nargs='+', type=int , default = [15817], help='''input dimension(s) separated in space the output will be a list, decide which embedding types to use. 
                        If len of 1, then  1 embedding; len of 3, embedding medical, diagnosis and others separately (3 embeddings) [default:[15817]]''')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training, validation or test [default: 128]')
    parser.add_argument('--eps', type=float, default=10**-8, help='term to improve numerical stability [default: 0.00000001]')
    parser.add_argument('--preTrainEmb', type= str, default='', help='path to pretrained embeddings file. [default:'']')

    # Hyperparameter optimization parameters
    parser.add_argument("--n_trials", type=int, help="Number of hyperparameter optimizations to run [default: 1 = running based on time_optim instead]", default=1)
    parser.add_argument("--train_epochs", type=int, help="Number of hyperparameter optimizations to run [default: 10]", default=100)
    parser.add_argument("--time_optim", type=int, help="How long to run hyperparameter optimizations.", default=300)
    parser.add_argument("--refit", type=int, help="Whether to rerun the hyperparameter optimization", default=1)
    parser.add_argument('--time', type=int, default=0, help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('--bii', type=int, default=0, help='indicator of whether Bi-directin is activated. [default: False]')

    # Final model fitting and evaluation
    parser.add_argument("--skip_model_fit", type=int, help="Whether to rerun the final model fitting, or load a prior model fit.", default=0)
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics.", default=500)
    parser.add_argument("--min_batch_cases", type=int, help="Minimum number of cases in a batch", default=0)

    args = parser.parse_args()
    return(args)

#do the main file functions and runs 
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    
    study_name = args.model_name + "_" + args.cell_type + "_" + args.pred_descriptor
    if args.min_batch_cases>0: study_name += "_min"+str(args.min_batch_cases)
    if args.bii == 1 and args.model_name != "TLSTM": study_name += "_bii"
    if args.time == 1 and args.model_name != "TLSTM": study_name += "_time"
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
    print(colored("\nLoading and preparing data...", 'green'))
    train_mbs, valid_mbs, test_mbs = get_mbs_from_files(args.file_path,
                                                        args.file_name_start,
                                                        args.model_name,
                                                        args.batch_size,
                                                        args.min_batch_cases)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Model loding                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #        
    if args.simple_fit == 0 and args.skip_model_fit == 0:
        base_params = {"cell_type": args.cell_type,
                       "input_size": args.input_size,
                       "bii": args.bii,
                       "time": args.time,
                       "preTrainEmb": args.preTrainEmb,
                       "early_stop": args.early_stop,
                       "epochs": args.train_epochs,
                       "model_name": args.model_name,
                       "eps": args.eps,
                       "eval_metric": "logloss"}
        best_params = run_optuna_optim(train=train_mbs, 
                                       valid=valid_mbs, 
                                       test=test_mbs, 
                                       lab_name=args.lab_name,
                                       refit=args.refit,
                                       time_optim=args.time_optim,
                                       n_trials=args.n_trials,
                                       study_name=study_name,
                                       res_dir=args.res_dir,
                                       model_type="torch",
                                       model_fit_date=args.model_fit_date,
                                       base_params=base_params)
    else:
        base_params = {"cell_type": args.cell_type,
                       "input_size": args.input_size,
                       "bii": args.bii,
                       "time": args.time,
                       "preTrainEmb": args.preTrainEmb,
                       "early_stop": args.early_stop,
                       "epochs": args.train_epochs,
                       "model_name": args.model_name,
                       "eps": args.eps,
                       "eval_metric": "logloss"}
        best_params = {'embed_dim_exp': 11, 
                       'hidden_size_exp': 11, 
                       'dropout_r': 1,
                       'L2': 1e-5, 
                       'lr': 11, 
                       'optimizer': 'adagrad'}

    if args.skip_model_fit == 0:
        timer = Timer()

        print(best_params)
        ehr_model = get_model(model_name=base_params["model_name"],
                                embed_dim_exp=best_params["embed_dim_exp"],
                                hidden_size_exp=best_params["hidden_size_exp"],
                                n_layers=1,
                                dropout_r=best_params["dropout_r"],
                                cell_type=base_params["cell_type"],
                                bii=base_params["bii"],
                                time=base_params["time"],
                                preTrainEmb=base_params["preTrainEmb"],
                                input_size=base_params["input_size"],
                                final_embed_dim_exp=best_params["final_embed_dim_exp"])   
        optimizer = get_torch_optimizer(ehr_model, 
                                        base_params["eps"],
                                        best_params["lr"], 
                                        best_params["L2"],
                                        best_params["optimizer"])
        if torch.cuda.is_available(): ehr_model = ehr_model.cuda() 
        try:
            _, _, best_model = epochs_run(50, 
                                          train = train_mbs, 
                                          valid = valid_mbs, 
                                          test = test_mbs, 
                                          model = ehr_model, 
                                          optimizer = optimizer,
                                          model_name = args.model_name, 
                                          early_stop = 5,
                                          model_out=out_model_dir + "final_model")
        except KeyboardInterrupt:
            print(colored('-' * 89, 'green'))
            print(colored('Exiting from training early','green'))
    
    best_model = torch.load(out_model_dir + "final_model.pth", weights_only=False)
    best_model.load_state_dict(torch.load(out_model_dir + "final_model.st"))
    best_model.eval()
    print(colored("\nFinal fitting done!", 'green'))
    print(timer.get_elapsed())

#    train_mbs = list(tqdm(EHRdataloader(train, batch_size = args.batch_size, packPadMode = pack_pad, shuffle=False)))
    out_data = get_out_data(best_model, 
                            train_mbs, 
                            valid_mbs, 
                            test_mbs, 
                            args.file_path_labels,
                            args.goal)
    out_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")  
    # crnt_report = create_report(best_model, out_data, display_scores=["logloss", "aucpr"], metric="logloss")
    # pickle.dump(crnt_report, open(out_model_dir + "report_" + get_date() + ".pkl", "wb"))  
    
    save_all_report_plots(out_data=out_data,
                          out_plot_path=out_plot_path,
                          out_down_path=out_down_path)

    # eval_metrics = get_all_eval_metrics(data=out_data, 
    #                                     plot_path=out_plot_path,
    #                                     down_path=out_down_path,
    #                                     n_boots=args.n_boots,
    #                                     train_type="bin")
    # if args.save_csv == 1:
    #     eval_metrics.filter(pl.col("F1").is_not_null()).write_csv(out_down_path + "evals_" + get_date() + ".csv", separator=",")
