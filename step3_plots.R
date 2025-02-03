library(readr)
source("/home/ivm/valid/scripts/utils.R")

in_dir <- "/home/ivm/valid/data/processed_data/step3/"
in_file_date <- "2024-10-18"
lab_name = "krea"
# FINNGENID, EVENT_AGE, DATE, VALUE, ABNORM (FinnGen), ABNORM_CUSTOM (Reference ranges), EDGE (helper 1: last measurement, 0: first measurement, 2: second to last measurement)
data <- readr::read_delim(paste0(in_dir, lab_name, "_", in_file_date, ".csv"))
# FINNGENID, SEX, + other metadata
metadata <- readr::read_delim(paste0(in_dir, lab_name, "_", in_file_date, "_meta.csv"))

#### Number of diagnoses
library(ggplot2)
metadata <- dplyr::filter(metadata, N_YEAR > 1)
n_measure <- metadata %>% group_by(N_MEASURE) %>% reframe(N_INDV=n()) %>% filter(N_INDV >= 5)
metadata %>% pull(N_MEASURE) %>% summary()
source("/home/ivm/valid/scripts/utils.R")
ggplot(n_measure, aes(x=N_MEASURE, y=N_INDV)) +
  geom_col(width=1.5) +
  theme_bar(line_size=1, show_x=TRUE) +
  scale_y_continuous(breaks=seq(1, 300000, 5000), labels=function(x){so_formatter(x)}) +
  scale_x_continuous(breaks = c(2, 25, 50, 75, 100)) +
  coord_cartesian(xlim=c(0,100)) +
  labs(x="Number of measurements", y="Number of individuals")

ggplot(metadata, aes(x=N_YEAR)) +
  geom_histogram()
