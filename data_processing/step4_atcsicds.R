.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")

library(readr)
library(dplyr)
library(optparse)

####### Setting up parser
args_list <- list(
  make_option(c("--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step5_data/data-diag/", help="Path to results directory."),
  make_option(c("--out_file_name"), action="store", type="character", default="",help="Identifer for out file name."),
  make_option(c("--col_name"), action="store", type="character", default="ICD_THREE",help="Name of column of predictor."),
  make_option(c("--file_path_preds"), action="store", type="character", default="/home/ivm/valid/data/extra_data/data/processed_data/step1/atcs_r12_2025-02-04_min1pct_sum_onttop_2025-02-18.csv", help="Full path to data."),
  make_option(c("--dir_path_labels"), type="character", action="store", help="Path to directory containing the label data."),
  make_option(c("--file_name_labels"), action="store", type="character", default="hba1c_d1_2025-02-10_data-diag_2025-02-17",help="File name of label file, without the '.csv'"),
  make_option(c("--time"), action="store", type="integer", default=0,help="Whether to filter for current and not historical data. -1 = historical, 0 = all, 1 = current"),
  make_option(c("--bin_count"), action="store", type="integer", default=0,help="Whether to count number of occurance or stay binary observed/not observed."),
  make_option(c("--months_before"), action="store", type="integer", default=0, help="Months to add before start of measurements for a buffer."),
  make_option(c("--start_year"), action="store", type="integer", default=0, help="what year to start the data from. Ignored if 0.")
)
parser <- OptionParser(option_list=args_list, add_help_option=FALSE)
args <- parse_args(parser, positional_arguments=0)$options
print(args)
date = Sys.Date()

####### Getting data for predictors
preds_data <- readr::read_delim(args$file_path_preds) # Predictors data

####### Getting information for historical or current data
# historical - all predictor information only collected before first measurement -X months
# current - all predictor information only collected after first measurement -X months
get_time_period_data <- function(time, 
                                 all_data_file_path, 
                                 preds_data, 
                                 months_before) {
    # Need date of first measurement of each individual to filter historical or current data
    start_dates <- read::read_delim(all_data_file_path) 
    start_dates <- dplyr::group_by(start_dates, FINNGENID) %>% 
                            dplyr::arrange(DATE) %>% slice(1L) %>% # Date of first recorded measurement for each individual
                            dplyr::rename(DATA_START_DATE=DATE) %>% 
                            dplyr::select(FINNGENID, DATA_START_DATE) %>%  
                            dplyr::ungroup()
    # Start date of information X-months before first measurement
    start_dates <- dplyr::mutate(start_dates , DATA_START_DATE=DATA_START_DATE%m+%months(-args$months_before))
    # Adding info to predictor data
    preds_data <- dplyr::left_join(preds_data, start_dates, by="FINNGENID")
    # Filtering out only current or historical data
    if(args$time == 1) preds_data <- dplyr::filter(preds_data, DATE>=DATA_START_DATE) %>% dplyr::select(-DATA_START_DATE)
    if(args$time == -1) preds_data <- dplyr::filter(preds_data, DATE<=DATA_START_DATE) %>% dplyr::select(-DATA_START_DATE)
    return(preds_data)
}
if(args$time != 1) preds_data = get_time_period_data(args$time, paste0(args$file_path_preds, args$dir_path_labels, ".csv"), preds_data, args$months_before) 

####### Getting information about start of prediction period = end of collection for predictors data
end_dates <- readr::read_delim(paste0(args$file_path_preds, args$dir_path_labels, "_labels.csv")) 
end_dates <- end_dates %>% dplyr::select(FINNGENID, START_DATE) %>% dplyr::mutate(START_DATE=as.Date(START_DATE))
# Adding info to predictor data
preds_data <- dplyr::left_join(preds_data, end_dates, by="FINNGENID")
# Filtering out only data before start of prediction period
preds_data <- dplyr::filter(preds_data, DATE < START_DATE)
# Filtering out data only after what we set as start year
if(args$start_year != 0){
  preds_data <- dplyr::filter(preds_data, lubridate::year(DATE)>=args$start_year)
}

####### Counting and getting info in wide data
# Counting
preds_wider <- preds_data %>% dplyr::group_by(FINNGENID, get(args$col_name)) %>% dplyr::reframe(N_CODE=n()) %>% dplyr::ungroup()
if(args$bin_count == 1) { # Making binary 0/1 observed
  preds_wider <- preds_wider %>% dplyr::mutate(N_CODE=ifelse(N_CODE>0, 1, 0))
}
# Making wide
preds_wider <- preds_wider %>% tidyr::pivot_wider(names_from=get(args$col_name), values_from=N_CODE, values_fill=0)

######## Adding info on when was last recorded Code
# Date of last code
last_date <- dplyr::select(preds_data, FINNGENID, DATE) %>% distinct() %>% dplyr::group_by(FINNGENID) %>% dplyr::arrange(FINNGENID, desc(DATE)) %>% slice(1L)
# Adding info to predictor data
preds_wider <- dplyr::left_join(preds_wider, last_date) %>% dplyr::rename(LAST_CODE_DATE=DATE)

######### Saving info
dir.create(args$res_dir, showWarnings=FALSE, recursive = TRUE)
out_file_path <- "atcs_"
if(args$col_name == "ICD_THREE") out_file_path <- "icds_"
if(args$out_file_name != "") out_file_path <- paste0(args$res_dir, out_file_path, args$out_file_name, "_", args$file_name, ".csv")
else out_file_path <- paste0(args$res_dir, out_file_path, args$file_name, ".csv")
readr::write_delim(preds_wider, out_file_path, delim=",")

