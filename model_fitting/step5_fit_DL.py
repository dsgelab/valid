try:
    from termcolor import colored
except:
    def colored(string, color): return(string)
########## Checked
sys.path.append(("/home/ivm/valid/scripts/pytorch_ehr/"))
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer
from torch_utils import get_mbs_from_files, get_model, get_torch_optimizer
from optuna_utils import run_optuna_optim
from utils_final import epochs_run, get_out_data
from model_fit_utils import save_all_report_plots
from model_eval_utils import eval_subset, create_report

# Logging and input
import argparse
import logging
logger = logging.getLogger(__name__)

# General
import torch
import pickle
from io import open
import sys


def get_parser_arguments():
    #this is where you define all the things you wanna run in your main file
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')
    
    # Data paths
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument('--file_path', type = str, required=True, help='the path to the folders with pickled file(s)')
    parser.add_argument('--files', nargs='+', default = ['hf.train'], help='''the name(s) of pickled file(s), separtaed by space. so the argument will be saved as a list 
                        If list of 1: data will be first split into train, validation and test, then 3 dataloaders will be created.
                        If list of 3: 3 dataloaders will be created from 3 files directly. Please give files in this order: training, validation and test.''')
   
    # Extra info
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--model_fit_date", type=str, help="Original date of model fitting.", default="")
    # Model fitting best_params
    parser.add_argument('--model_name', type=str, default='RNN',choices= ['RNN','DRNN','QRNN','TLSTM','LR','RETAIN'], help='choose from {"RNN","DRNN","QRNN","TLSTM","LR","RETAIN"}') 
    parser.add_argument('--simple_fit', type=int, default=0, help="Runs a simple fit of the model with no hyperparameter tuning. [default: 0]") 
    parser.add_argument('--skip_model_fit', type=int, default=0, help="Skip model fitting and load the model from the file. [default: 0]") 
    parser.add_argument("--early_stop", type=int, help="Early stopping for the final fitting round. Currently, early stopping fixed at 5 for hyperparameter optimization.", default=5)
    # Preset best_params
    parser.add_argument('--cell_type', type = str, default = 'GRU', choices=['RNN', 'GRU', 'LSTM'], help='For RNN based models, choose from {"RNN", "GRU", "LSTM", "QRNN" (for QRNN model only)}, "TLSTM (for TLSTM model only')
    parser.add_argument('-input_size', nargs='+', type=int , default = [15817], help='''input dimension(s) separated in space the output will be a list, decide which embedding types to use. 
                        If len of 1, then  1 embedding; len of 3, embedding medical, diagnosis and others separately (3 embeddings) [default:[15817]]''')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training, validation or test [default: 128]')
    parser.add_argument('--eps', type=float, default=10**-8, help='term to improve numerical stability [default: 0.00000001]')
    parser.add_argument('--preTrainEmb', type= str, default='', help='path to pretrained embeddings file. [default:'']')

    # Hyperparameter optimization parameters
    parser.add_argument("--n_trials", type=int, help="Number of hyperparameter optimizations to run [default: 1 = running based on time_step1 instead]", default=1)
    parser.add_argument("--time_optim", type=int, help="Number of seconds to run hyperparameter optimizations for, instead of basing it on the number of traisl. [run when n_trials=1]", default=300)
    parser.add_argument("--refit", type=int, help="Whether to rerun the hyperparameter optimization", default=1)
    parser.add_argument('--time', type=int, default=0, help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('--bii', type=int, default=0, help='indicator of whether Bi-directin is activated. [default: False]')

    # Extra loading best_params
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.1)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)

    # Final model fitting and evaluation
    parser.add_argument("--skip_model_fit", type=int, help="Whether to rerun the final model fitting, or load a prior model fit.", default=0)
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics.", default=500)

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
                                                        args.file_names,
                                                        args.model_name,
                                                        args.valid_pct,
                                                        args.test_pct,
                                                        args.batch_size)
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
                       "epochs": args.epochs,
                       "model_type": "torch"}
        best_params = run_optuna_optim(train=train_mbs, 
                                       valid=valid_mbs, 
                                       test=test_mbs, 
                                       lab_name=args.lab_name,
                                       refit=args.refit,
                                       time_optim=args.time_step1,
                                       n_trial=args.n_trials,
                                       study_name=study_name,
                                       res_dir=args.res_dir,
                                       model_fit_date=args.model_fit_date,
                                       base_params=base_params)
    else:
        best_params = {'embed_dim_exp': 7, 
                       'hidden_size_exp': 7, 
                       'dropout_r': 4,
                       'L2': 0.0009451300184944492, 
                       'lr': 8, 
                       'optimizer': 'adamax'}

    if args.skip_model_fit == 0:
        print(best_params)
        ehr_model = get_model(model_name=base_params["model_name"],
                                embed_dim_exp=best_params["embed_dim_exp"],
                                hidden_size_exp=best_params["hidden_size_exp"],
                                n_layers=best_params["n_layers"],
                                dropout_r=best_params["dropout_r"],
                                cell_type=base_params["cell_type"],
                                bii=base_params["bii"],
                                time=base_params["time"],
                                preTrainEmb=base_params["preTrainEmb"],
                                input_size=base_params["input_size"])   
        optimizer = get_torch_optimizer(ehr_model, 
                                        base_params["eps"],
                                        best_params["lr"], 
                                        best_params["L2"],
                                        best_params["optimizer_name"])
        if torch.cuda.is_available(): ehr_model = ehr_model.cuda() 
        try:
            _, _, best_model = epochs_run(args.epochs, 
                          train = train_mbs, 
                          valid = valid_mbs, 
                          test = test_mbs, 
                          model = ehr_model, 
                          optimizer = optimizer,
                          shuffle = True, 
                          model_name = args.model_name, 
                          early_stop = 10,
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
                            args.file_path_labels)
    out_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")  
    crnt_report = create_report(best_model, out_data, display_scores=["logloss", "aucpr"], metric="logloss")
    pickle.dump(crnt_report, open(out_model_dir + "report_" + get_date() + ".pkl", "wb"))  

    save_all_report_plots(out_data=out_data,
                          out_plot_path=out_plot_path,
                          out_down_path=out_down_path)

    eval_metrics, all_conf_mats = eval_subset(data=out_data, 
                                              y_pred_col="ABNORM_PREDS", 
                                              y_cont_pred_col="ABNORM_PROBS", 
                                              y_goal_col="TRUE_ABNORM", 
                                              y_cont_goal_col="TRUE_VALUE",
                                              plot_path=out_plot_path,
                                              down_path=out_down_path,
                                              subset_name="all",
                                              n_boots=args.n_boots,
                                              train_type="bin")
    if args.save_csv == 1:
        eval_metrics.filter(pl.col("F1").is_not_null()).write_csv(out_down_path + "evals_" + get_date() + ".csv", separator=",")
