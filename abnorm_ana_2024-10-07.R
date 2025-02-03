library(readr)
library(dplyr)
library(lubridate)

get_abnorm_lens <- function(data) {
  data <- data %>% 
    # No missing abnormality
    dplyr::filter(!is.na(FIRST_ABNORM_DATE), !is.na(ABNORM)) %>% 
    arrange(FINNGENID, DATE) %>% 
    select(FINNGENID, DATE, VALUE, ABNORM) %>%
    # 
    dplyr::group_by(FINNGENID) %>% 
    # Cannot define length for single value individuals
    dplyr::filter(n() > 1) %>%
    dplyr::mutate(PREV_ABNORM=lag(ABNORM),
                  # Very start of the period has an extra value that it is not the same as ABNORM
                  PREV_ABNORM=ifelse(is.na(PREV_ABNORM), 2, PREV_ABNORM),
                  START=ifelse(ABNORM!=PREV_ABNORM, "START", NA),
                  START_DATE=ifelse(!is.na(START), DATE, NA)) 
    # Start filling start date column for first rows for block
    data$START_DATE <- as.Date(NA)
    data$START_DATE[!is.na(data$START)] <- as.Date(data$DATE[!is.na(data$START)])

    # Fill consequent rows of block with start date from before recursive until all filled
    while(sum(is.na(data %>% pull(START_DATE))) > 0) {
      data$START_DATE[is.na(data$START_DATE)] <- as.Date(lag(data$START_DATE)[is.na(data$START_DATE)])
      print(sum(is.na(data %>% pull(START_DATE))))
    }
  
    data <- dplyr::group_by(data, FINNGENID, ABNORM, START_DATE) %>%
      # Cannot define period when less than one measurement available
      dplyr::filter(n() >= 2) %>%
      # Length of measurements between first and last of that abnormality
      dplyr::reframe(N_MEASURE=n(), DIFF=time_length(min(DATE)%--%max(DATE), "days")) %>%
      dplyr::distinct()
    
  return(data)
}

#egfr_abnorm_lens <- get_abnorm_lens(egfr_data)
#readr::write_delim(egfr_abnorm_lens, "/home/ivm/valid/data/processed_data/step3/krea_2024-10-18_abnormlens_fg.csv", delim=",")

#egfr_abnorm_custom_lens <- get_abnorm_lens(egfr_data %>% select(-ABNORM, -FIRST_ABNORM_DATE) %>% rename(ABNORM=ABNORM_CUSTOM, FIRST_ABNORM_DATE=FIRST_ABNORM_CUSTOM_DATE))
#readr::write_delim(egfr_abnorm_custom_lens, "/home/ivm/valid/data/processed_data/step3/krea_2024-10-18_abnormlens_custom.csv", delim=",")

something <- function(data, abnorm=0) {
  data_tab <- table(round(data %>% filter(ABNORM == abnorm) %>% pull(DIFF)))
  data_tib <- tibble(DIFF=as.numeric(names(data_tab)), N=as.numeric(data_tab))
  
  print(min(data_tib %>% filter(N>= 5) %>% pull(DIFF), na.rm=TRUE))
  print(max(data_tib %>% filter(N>= 5) %>% pull(DIFF), na.rm=TRUE))
  print(summary(round(data %>% filter(ABNORM == abnorm) %>% pull(DIFF))))
}

egfr_data <- readr::read_delim("/home/ivm/valid/data/processed_data/step3/krea_2024-10-17.csv")
egfr_meta <- readr::read_delim("/home/ivm/valid/data/processed_data/step3/krea_2024-10-17_meta.csv")
egfr_data <- dplyr::left_join(egfr_data, egfr_meta)
egfr_abnorm_lens <- get_abnorm_lens(egfr_data)
something(egfr_abnorm_lens, abnorm=1)
something(egfr_abnorm_lens, abnorm=0)

data_diag <- egfr_abnorm_lens %>% filter(ABNORM == 1, DIFF > 90) %>% arrange(FINNGENID, START_DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 
data_diag_custom <- egfr_abnorm_custom_lens %>% filter(ABNORM == 1, DIFF > 90) %>% arrange(FINNGENID, START_DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 
data_diag <- dplyr::full_join(data_diag %>% ungroup() %>% select(FINNGENID, START_DATE) %>% rename(DATA_DIAG_DATE=START_DATE), data_diag_custom %>% ungroup() %>% select(FINNGENID, START_DATE) %>% rename(DATA_DIAG_CUSTOM_DATE=START_DATE))

readr::write_delim(data_diag, "/home/ivm/valid/data/processed_data/step3/krea_2024-10-18_datadiags.csv", delim=",")
egfr_first_longest <- egfr_abnorm_lens %>% arrange(FINNGENID, DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 

something(egfr_first_longest, abnorm=0)

#test %>% group_by(ABNORM) %>% reframe(mean=mean(DIFF, na.rm=TRUE),median=median(DIFF, na.rm=TRUE))

tsh_data <- readr::read_delim("/home/ivm/valid/data/processed_data/step2/tsh_2024-10-07.csv")
tsh_abnorm_lens <- get_abnorm_lens(tsh_data)
tsh_first_longest <- tsh_abnorm_lens %>% arrange(FINNGENID, DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 

something(tsh_first_longest, 1)
something(tsh_first_longest, 0)
something(tsh_first_longest, -1)

tsh_ca_data <- readr::read_delim("/home/ivm/valid/data/processed_data/step2/tsh_ca_2024-10-08.csv")
tsh_dif_abnorm_lens <- get_abnorm_lens(tsh_ca_data)
tsh_dif_first_longest <- tsh_dif_abnorm_lens %>% arrange(FINNGENID, DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 

tsh_ca_data$ABNORM[tsh_ca_data$ABNORM == 2] <- 1
tsh_ca_abnorm_lens <- get_abnorm_lens(tsh_ca_data)
tsh_ca_first_longest <- tsh_ca_abnorm_lens %>% arrange(FINNGENID, DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 

something(tsh_ca_first_longest, 1)
something(tsh_ca_first_longest, 0)
something(tsh_ca_first_longest, -1)

hba_data <- readr::read_delim("/home/ivm/valid/data/processed_data/step2/hba1c_2024-10-07.csv")
hba_abnorm_lens <- get_abnorm_lens(hba_data)
hba_first_longest <- hba_abnorm_lens %>% arrange(FINNGENID, DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 


hba_ca_data <- readr::read_delim("/home/ivm/valid/data/processed_data/step2/hba1c_2024-10-07.csv")
hba_ca_abnorm_lens <- get_abnorm_lens(hba_ca_data)
hba_ca_first_longest <- hba_ca_abnorm_lens %>% arrange(FINNGENID, DATE, ABNORM) %>% group_by(FINNGENID, ABNORM) %>% slice(1L) 

something(hba_ca_first_longest, 1)
something(hba_ca_first_longest, 0)

all_longest <- rbind(tsh_first_longest %>% mutate(LAB="TSH"),
                     tsh_dif_first_longest %>% mutate(LAB="TSH Diff"), 
                     tsh_ca_first_longest %>% mutate(LAB="TSH CA"), 
                     egfr_abnorm_lens %>% mutate(LAB="eGFR"), 
                     hba_first_longest %>% mutate(LAB="HbA1c"))

all_longest <- all_longest %>% mutate(ABNORM=case_when(ABNORM == -1 ~ "Low", 
                                          ABNORM == 0 ~ "Normal",
                                          ABNORM == 1 & LAB != "TSH CA" ~ "High",
                                          ABNORM == 1 & LAB == "TSH CA" ~ "Grey", 
                                          ABNORM == 2 ~ "High"))
all_longest$ABNORM <- factor(all_longest$ABNORM, levels=c("Normal", "Low", "High", "Grey"))
library(ggplot2)
ggplot(all_longest %>% filter(LAB %in% c("eGFR", "HbA1c", "TSH")), aes(x=ABNORM, y=DIFF, fill=LAB)) + 
  geom_boxplot(outlier.shape=NA, position =  position_dodge2(width = 0.75, preserve = "single")) +
  theme_custom() +
  scale_fill_manual(values=custom_colors_brewer(5)) +
  scale_y_continuous(limits=c(0,1000))

#' Custom color selection 
#' 
#' Based on https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=10
#' 
#' @param N_colors The number of colors
#' 
#' @author Kira E. Detrois
#' 
#' @export 
custom_colors_brewer <- function(N_colors) {
  if(N_colors == 1)
    c("#000000")
  else if(N_colors == 2) {
    c("#D5694F", "#29557B")
  } else if(N_colors == 3)  {
    c("#D5694F", "#29557B", "#EAB034")
  } else if(N_colors == 4) {
    c("#D5694F", "#29557B", "#EAB034", "#748AAA")
  } else if(N_colors == 5) {
    c("#D5694F", "#29557B", "#EAB034", "#748AAA", "#CCB6AF")
  } else if(N_colors == 10) {
    c("#D5694F", "#29557B", "#EAB034", "#748AAA", "#CCB6AF", "#841C26", "#7D7C7F", "#FBCF9D",  "#7BA05B", "#588986")
  } else if(N_colors == 17) {
    c("#841C26", "#B53389", "#C6878F", "#A81C07", "#D5694F", "#FBCF9D", "#59260B", "#CCB6AF", "#7D7C7F", "#91A3B0",
      "#3C4E2D", "#7BA05B", "#9BB59B", "#588986","#29557B","#748AAA", "#ADD8E6")
  } else if(N_colors == 18) {
    c("#841C26", "#B53389", "#C6878F", "#A81C07", "#D5694F", "#FBCF9D", "#FBCF9D", "#CCB6AF", "#7D7C7F", "#91A3B0",
      "#3C4E2D", "#7BA05B", "#9BB59B", "#588986","#29557B","#748AAA", "#ADD8E6", "#D6ECFF")
  }
}
##################### THEMES ####################
theme_custom <- function(base_size = 18,
                         legend_pos = "bottom",
                         plot_top_margin = -30,
                         axis_x_bottom_margin=0,
                         axis_y_left_margin=0,
                         legend_box_spacing=1,
                         line_size=1.5) {
  ggplot2::theme_minimal(base_size = base_size) %+replace%
    ggplot2::theme(
      text=ggplot2::element_text(colour="black"),
      # Titles
      plot.title=ggplot2::element_text(hjust=0, margin=margin(t=plot_top_margin, b=5), size=base_size),
      plot.subtitle=ggplot2::element_text(hjust=0, size=base_size*0.9,  margin=margin(t=plot_top_margin, b=5), face="bold"),
      plot.caption=ggplot2::element_text(size=base_size*0.6, hjust=0, margin=margin(t=10)),
      # Facet grid / wrap titles
      strip.text = ggplot2::element_text(hjust=0, face="bold", size=base_size*0.8, margin=margin(b=5)),
      # Legend
      legend.title=ggplot2::element_blank(),
      legend.position = legend_pos,
      legend.text=ggplot2::element_text(size=base_size*0.8),
      legend.key.spacing.y=grid::unit(0.5, "lines"),
      legend.margin = ggplot2::margin(legend_box_spacing, legend_box_spacing, legend_box_spacing, legend_box_spacing),
      # Axes
      axis.title=ggplot2::element_text(size=base_size*0.8),
      axis.text = ggplot2::element_text(size=base_size*0.75),
      axis.title.x = ggplot2::element_text(margin=margin(t=5, b=axis_x_bottom_margin)),
      axis.title.y = ggplot2::element_text(margin=margin(r=5), l=axis_y_left_margin, angle=90),
      # Other settings
      panel.border = ggplot2::element_blank(),
      panel.background = ggplot2::element_rect(colour=NA, fill=NA),
      # Grid settings
      panel.grid.minor.x = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(colour="black", linewidth=line_size-0.5*line_size, linetype=2),
      panel.grid.minor.y = ggplot2::element_line(colour = "black", linewidth=line_size-0.75*line_size, linetype=2),
    )
}
