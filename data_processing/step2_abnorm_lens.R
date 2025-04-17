.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")


library(optparse)
library(readr)
library(dplyr)
library(lubridate)
library(arrow)


args_list <- list(
  make_option(c("-r", "--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step3_meta/",help="Path to results directory."),
  make_option(c("-f", "--file_path"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step1_clean/", help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end"),
  make_option(c("-n", "--file_name"), type="character", action="store", default="krea",help="Readable name of the measurement value."),
  make_option(c("-l", "--lenient_abnorm"), type="character", action="store", default="krea",help="Close to abnormal values marked with 0.5 can interrupt the block of abnormality.")
)

parser <- OptionParser(option_list=args_list)
args <- parse_args(parser, positional_arguments=0)$options
print(args)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Getting block lengths                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
get_abnorm_start_dates <- function(data) {
  data <- data %>% 
    # # # # # # # Selecting and sorting  # # # # # # # # # # # # 
    dplyr::filter(!is.na(ABNORM_BIN)) %>% 
    dplyr::arrange(FINNGENID, DATE) %>% 
    dplyr::select(FINNGENID, DATE, VALUE, ABNORM_BIN) %>%
    dplyr::mutate(DATE=as.Date(DATE)) %>%

    # # # # # # # For each individual  # # # # # # # # # # # # 
    dplyr::group_by(FINNGENID) %>% 
    # Cannot define length for single value individuals
    dplyr::filter(n() > 1) %>%
    dplyr::mutate(PREV_ABNORM=lag(ABNORM_BIN), 
                  # Very start of the period has an extra value that it is not the same as ABNORM
                  PREV_ABNORM=ifelse(is.na(PREV_ABNORM), -1, PREV_ABNORM),
                  # Marking start of abnormality section 
                  START=ifelse(ABNORM_BIN!=PREV_ABNORM, "START", NA)) 

  # # # # # # # Getting start dates # # # # # # # # # # # # # #
  # Need to do this manual because ifelse would destroy the Date format
  data$START_DATE <- as.Date(NA)
  data$START_DATE[!is.na(data$START)] <- as.Date(data$DATE[!is.na(data$START)])
  # Fill consequent rows of block with start date from before recursive until all filled
  while(sum(is.na(data %>% pull(START_DATE))) > 0) {
    data$START_DATE[is.na(data$START_DATE)] <- as.Date(lag(data$START_DATE)[is.na(data$START_DATE)])
  }
  
  return(data)
}

get_block_lengths <- function(data) {
  dplyr::group_by(data, FINNGENID, ABNORM_BIN, START_DATE) %>%
                      # Cannot define period when less than one measurement available
                      dplyr::filter(n() > 1) %>%
                      # Length of measurements between first and last of that abnormality
                      dplyr::reframe(N_MEASURE=n(), 
                                     DIFF=time_length(min(DATE)%--%max(DATE), "days")) %>%
                      dplyr::distinct()
}

data <- tibble::as_tibble(arrow::read_parquet(paste0(args$file_path, args$file_name, ".parquet")))
# It is easiest to just remove the 0.5 sections from the data at this point.
# Would need a different solution if interested in how long those sections are
if(args$lenient_abnorm == 1) {
  data <- dplyr::filter(data, ABNORM_CUSTOM != 0.5)
}
# Create binary abnormality variable - in case there is further 
# partition between slight abnormal (1) and higher abnormal (i.e. 2)
data <- dplyr::mutate(data, ABNORM_BIN=ifelse(ABNORM_CUSTOM==0, 0, 1))
prep_data <- get_abnorm_start_dates(data) 
all_times <- get_block_lengths(prep_data)

out_file_path <- paste0(args$res_dir, args$file_name, "_", Sys.Date())
arrow::write_parquet(all_times, paste0(out_file_path, "_alltimes.parquet"))
