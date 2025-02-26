

.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")

library(readr)
library(dplyr)
library(tidyr)
library(optparse)

args_list <- list(
  make_option(c("-r", "--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step5_data/data-diag/",
              help="Path to results directory."),
  make_option(c("-f", "--file_path_lab"), action="store", type="character", default="/home/ivm/valid/data/extra_data/data/processed_data/step0/kanta_lab_min1pct_18702026_2025-02-20.csv",
              help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end"),
  make_option(c("-s", "--file_path"), type="character", action="store", default="krea",help="Readable name of the measurement value."),
  make_option(c("-d", "--file_name"), action="store", type="character", default="hba1c_d1_2025-02-10_data-diag_2025-02-17",help="Date of original processed file."),
  make_option(c("-o", "--out_file_name"), action="store", type="character", default="",help="idnetifier for out file."),
  make_option(c("-l", "--lab_name"), action="store", type="character", default="",help="idnetifier for out file.")
  
)

parser <- OptionParser(option_list=args_list, add_help_option=FALSE)
args <- parse_args(parser, positional_arguments=0)$options
print(args)

date = Sys.Date()
lab_data <- readr::read_delim(args$file_path_lab) %>% 
              dplyr::select(FINNGENID, APPROX_EVENT_DATETIME, OMOP_CONCEPT_ID, MEASUREMENT_VALUE_HARMONIZED) %>%
              dplyr::mutate(APPROX_EVENT_DATETIME=as.Date(APPROX_EVENT_DATETIME))

file_path_labels <- paste0(args$file_path, args$file_name, "_labels.csv")
end_dates <- readr::read_delim(file_path_labels)
lab_data <- dplyr::filter(lab_data, FINNGENID %in% end_dates$FINNGENID)
if(args$lab_name == "krea") {
  lab_data <- dplyr::filter(lab_data, !(OMOP_CONCEPT_ID %in% c("40764999", "3020564")))
}
if(args$lab_name == "hba1c") {
  lab_data <- dplyr::filter(lab_data, !(OMOP_CONCEPT_ID %in% c("3004410")))
}
# Data in at least 1% of individuals
end_dates <- end_dates %>% dplyr::select(FINNGENID, START_DATE)
lab_data <- dplyr::left_join(lab_data, end_dates, by="FINNGENID")
lab_data <- dplyr::mutate(lab_data, START_DATE=as.Date(START_DATE))
lab_data <- dplyr::filter(lab_data, APPROX_EVENT_DATETIME < START_DATE)

# Count
lab_data <- lab_data %>% 
                  dplyr::group_by(FINNGENID, OMOP_CONCEPT_ID) %>% 
                  dplyr::arrange(desc(APPROX_EVENT_DATETIME)) %>%
                  dplyr::reframe(MEAN=mean(MEASUREMENT_VALUE_HARMONIZED),
                                 QUANT_25=quantile(MEASUREMENT_VALUE_HARMONIZED, 0.75),
                                 QUANT_75=quantile(MEASUREMENT_VALUE_HARMONIZED, 0.25)) 
print(lab_data)

## Adding info on when was last recorded ATC-code
lab_data <- lab_data %>% tidyr::pivot_longer(cols=c("MEAN", "QUANT_25", "QUANT_75"), names_to="STAT", values_to="VALUE")
print(lab_data)

lab_data <- lab_data %>% dplyr::mutate(NAME=paste0(OMOP_CONCEPT_ID,  "_",  STAT)) %>% dplyr::select(-OMOP_CONCEPT_ID, -STAT)
print(lab_data)

lab_data <- lab_data %>% tidyr::pivot_wider(names_from=NAME, values_from=VALUE)
print(lab_data)

dir.create(args$res_dir, showWarnings=FALSE, recursive = TRUE)

out_file_path <- ""
if(args$out_file_name != "") {
  out_file_path <- paste0(args$res_dir, "lab_", args$out_file_name, "_", args$file_name, ".csv")
} else {
  out_file_path <- paste0(args$res_dir, "lab_", args$file_name, ".csv")
}
readr::write_delim(lab_data, out_file_path, delim=",")
