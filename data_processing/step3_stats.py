
def get_crnt_stats(data, subset_name="All", subset_query="", abnorm_col="ABNORM", first_abnorm_col="FIRST_ABNORM_DATE", abnorm_type=1, abnorm_def="FG"):
    crnt_stats = pd.DataFrame({"STEP": ["N", "A", "N->A", "N->A(+D)", "N->A+D", "A2023(+D)", "FA2023(+D)"]})
    if subset_query != "": data = data.query(subset_query)
    # All normal
    grouped_data = data.groupby("FINNGENID")
    crnt_stats.loc[crnt_stats.STEP == "N","N_INDV"] = len(set(grouped_data.filter(lambda x: all(x.loc[:,abnorm_col] == 0)).FINNGENID))
    # Any abnormal
    abnorm_data = data.loc[data.loc[:,abnorm_col] == abnorm_type]
    crnt_stats.loc[crnt_stats.STEP == "A","N_INDV"] = len(set(abnorm_data.FINNGENID))
    # Abnormal 2023 + no diag before
    crnt_stats.loc[crnt_stats.STEP == "A2023(+D)","N_INDV"] = len(set(abnorm_data.query("DATE.dt.year == 2023 & (EITHER_DATE.dt.year > 2022 | EITHER_DATE.isnull())").FINNGENID))
    # First abnorm 2023 + no diag before
    first_abnorm_2023 = abnorm_data.loc[abnorm_data.loc[:,first_abnorm_col].dt.year > 2022]
    crnt_stats.loc[crnt_stats.STEP == "FA2023(+D)","N_INDV"] = len(set(first_abnorm_2023.query("(EITHER_DATE.dt.year > 2022 | EITHER_DATE.isnull())").FINNGENID))
    # Normal to abnormal (any normal before an abnormal)
    normal_data = data.loc[data.loc[:,abnorm_col] == 0]
    normal_abnormal_data = normal_data.loc[normal_data.DATE < normal_data.loc[:,first_abnorm_col]]
    crnt_stats.loc[crnt_stats.STEP == "N->A","N_INDV"] = len(set(normal_abnormal_data.FINNGENID))
    crnt_stats.loc[crnt_stats.STEP == "N->A(+D)","N_INDV"] = len(set(normal_abnormal_data.query("(DATE < EITHER_DATE | EITHER_DATE.isnull())").FINNGENID))
    crnt_stats.loc[crnt_stats.STEP == "N->A+D","N_INDV"] = len(set(normal_abnormal_data.query("DATE < EITHER_DATE").FINNGENID))

    crnt_stats.loc[:,"SUBSET"] = subset_name
    crnt_stats.loc[:,"ABNORM_DEF"] = abnorm_def
    crnt_stats.loc[:,"ABNORM_TYPE"] = abnorm_type
    return(crnt_stats)


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step3/")
    parser.add_argument("--file_path", type=str, help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end", default="/home/ivm/valid/data/processed_data/step2/")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--source_file_date", type=str, help="Date of file.", required=True)
parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)


    args = parser.parse_args()

    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
     
if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = args.lab_name + "_" + date
    file_path_data = args.file_path + args.lab_name + "_" + args.source_file_date + ".csv"
    file_path_meta = args.file_path + args.lab_name + "_" + args.source_file_date + "_meta.csv"
    log_file_name = args.lab_name + "_" + date_time
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, date_time)

    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")
    