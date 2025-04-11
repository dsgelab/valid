

.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")

library(readr)
library(dplyr)
library(tidyr)
library(arrow)
library(optparse)

####### Setting up parser
args_list <- list(
  make_option(c("--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step5_data/data-diag/", help="Path to results directory."),
  make_option(c("--file_path_lab"), action="store", type="character",  help="Full path to laboratory value data.", default="/home/ivm/valid/data/extra_data/processed_data/step1_clean/kanta_lab_min1pct_18702026_2025-02-20.csv"),
  make_option(c("--dir_path_labels"), type="character", action="store", help="Path to directory containing the label data."),
  make_option(c("--file_name_labels"), action="store", type="character", default="hba1c_d1_2025-02-10_data-diag_2025-02-17",help="File name of label file, without the '.csv'"),
  make_option(c("--out_file_name"), action="store", type="character", default="",help="idnetifier for out file."),
  make_option(c("--lab_name"), action="store", type="character", default="",help="Prediction lab name, for exclusion of predictors.")
)
parser <- OptionParser(option_list=args_list, add_help_option=FALSE)
args <- parse_args(parser, positional_arguments=0)$options
print(args)
date = Sys.Date()

####### Getting data for predictors
lab_data <- readr::read_delim(args$file_path_lab) %>% 
  dplyr::select(FINNGENID, APPROX_EVENT_DATETIME, OMOP_CONCEPT_ID, MEASUREMENT_VALUE_HARMONIZED) %>%
  dplyr::mutate(APPROX_EVENT_DATETIME=as.Date(APPROX_EVENT_DATETIME))
####### Getting information about start of prediction period = end of collection for predictors data
end_dates <- tibble::as_tibble(arrow::read_parquet(paste0(args$dir_path_labels, args$file_name_labels, "_labels.parquet")))
end_dates <- end_dates %>% dplyr::select(FINNGENID, START_DATE) %>% dplyr::mutate(START_DATE=as.Date(START_DATE))
# Adding info to predictor data
lab_data <- dplyr::left_join(lab_data, end_dates, by="FINNGENID")
# Filtering out only data before start of prediction period
lab_data <- dplyr::filter(lab_data, APPROX_EVENT_DATETIME < START_DATE)

######## Removing duplicate predictors
if(args$lab_name == "krea" | args$lab_name == "egfr") {
  lab_data <- dplyr::filter(lab_data, !(OMOP_CONCEPT_ID %in% c("40764999", "3020564")))
}
if(args$lab_name == "hba1c") {
  lab_data <- dplyr::filter(lab_data, !(OMOP_CONCEPT_ID %in% c("3004410")))
}
if(args$lab_name == "alatasat") {
  lab_data <- dplyr::filter(lab_data, !(OMOP_CONCEPT_ID %in% c("3006923", "3013721"))) #ALAT, ASAT
}
######### Stats for labs
lab_data <- lab_data %>% 
  dplyr::group_by(FINNGENID, OMOP_CONCEPT_ID) %>% 
  dplyr::reframe(MEAN=mean(MEASUREMENT_VALUE_HARMONIZED),
                 QUANT_25=quantile(MEASUREMENT_VALUE_HARMONIZED, 0.75),
                 QUANT_75=quantile(MEASUREMENT_VALUE_HARMONIZED, 0.25)) 
# Merging all info in single column
lab_data <- lab_data %>% tidyr::pivot_longer(cols=c("MEAN", "QUANT_25", "QUANT_75"), names_to="STAT", values_to="VALUE")
# Adding column with name for each
lab_data <- lab_data %>% dplyr::mutate(NAME=paste0(OMOP_CONCEPT_ID,  "_",  STAT)) %>% dplyr::select(-OMOP_CONCEPT_ID, -STAT)
# to then pivot it to wider format with each column name concept + stat
lab_data <- lab_data %>% tidyr::pivot_wider(names_from=NAME, values_from=VALUE)

######### Saving info
dir.create(args$res_dir, showWarnings=FALSE, recursive = TRUE)
out_file_path <- ""
if(args$out_file_name != "") {
  out_file_path <- paste0(args$res_dir, "labs_", args$out_file_name, "_", args$file_name, ".parquet")
} else {
  out_file_path <- paste0(args$res_dir, "labs_", args$file_name, ".parquet")
}
arrow::write_parquet(lab_data, out_file_path)
