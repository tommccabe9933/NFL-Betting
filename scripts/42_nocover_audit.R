# Audit: no-cover model data integrity
library(tidyverse); library(here); library(glmnet)
source(here("scripts", "00_project_setup.R"))
v3 <- read_rds(here("objects", "modeling_data_v4.rds"))

ff<-c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff","rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa","injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged","fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3","fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3","off_success_rate_roll5_diff","def_pass_epa_roll3_diff","fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff","fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5","fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5","dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff","off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff","def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
bn<-c("total_spread_script","high_total_big_spread","dog_cold_streak","dog_hot_streak","fade_dog_heat","qb_scheme_fit","defensive_style_clash","fav_expected_score","dog_expected_score","fav_pass_lean","dog_def_weakness")
sf<-setdiff(v3|>select(where(is.numeric),-season,-week)|>names(),c(ff,bn))
nc_target <- as.integer(v3$fav_cover == "no_cover")

cat("=== 1. TARGET ===\n")
cat(sprintf("Games: %d | Cover: %.1f%% | No-cover: %.1f%%\n", nrow(v3),
  mean(v3$fav_cover=="cover")*100, mean(v3$fav_cover=="no_cover")*100))
cat(sprintf("spread_fav: [%.1f, %.1f] %s\n\n", min(v3$spread_fav), max(v3$spread_fav),
  if(all(v3$spread_fav>0)) "OK" else "BUG"))

cat("=== 2. LEAKAGE ===\n")
leakers <- 0
for(f in sf){x<-v3[[f]];v<-!is.na(x);if(sum(v)<100)next
  cv<-tryCatch(cor(x[v],nc_target[v]),error=function(e)NA);if(is.na(cv))next
  if(abs(cv)>0.15){cat(sprintf("  LEAKER: %s cor=%+.3f\n",f,cv));leakers<-leakers+1}}
if(leakers==0) cat("  CLEAN\n")

cat("\n=== 3. TRAIN/TEST SEPARATION ===\n")
for(ts in c(2023,2024,2025)){
  tr<-v3|>filter(season<ts);te<-v3|>filter(season==ts)
  ov<-length(intersect(tr$game_id_norm,te$game_id_norm))
  cat(sprintf("  %d: train=%d test=%d overlap=%d %s\n",ts,nrow(tr),nrow(te),ov,if(ov==0)"OK"else"BUG"))}

cat("\n=== 4. GLMNET DIRECTION ===\n")
train<-v3|>filter(season<2023);test<-v3|>filter(season==2023)
x_tr<-train|>select(spread_fav,elo_diff)|>mutate(across(everything(),~replace_na(.x,0)))|>as.matrix()
x_te<-test|>select(spread_fav,elo_diff)|>mutate(across(everything(),~replace_na(.x,0)))|>as.matrix()
y_nc<-factor(train$fav_cover,levels=c("no_cover","cover"))
fit<-glmnet(x_tr,y_nc,family="binomial",alpha=0,lambda=0.01)
pred<-predict(fit,x_te,type="response")[,1]
p_nc<-1-pred
acc1<-mean((p_nc>0.5)==(test$fav_cover=="no_cover"))
acc2<-mean((pred>0.5)==(test$fav_cover=="no_cover"))
cat(sprintf("  1-pred direction: %.1f%% | pred direction: %.1f%% | %s\n",
  acc1*100,acc2*100,if(acc1>acc2)"1-pred CORRECT"else"FLIP BUG!"))

cat("\n=== 5. ROLLING LEAKAGE ===\n")
wk1<-v3|>filter(season==2025,week==1)|>head(1)
cat(sprintf("  2025 wk1 roll5: %s\n",ifelse(is.na(wk1$fav_off_epa_play_roll5),"NA (correct)","HAS VALUE")))
wk5<-v3|>filter(season==2025,week==5)|>head(1)
cat(sprintf("  2025 wk5 roll5: %s\n",ifelse(is.na(wk5$fav_off_epa_play_roll5),"NA (PROBLEM)",
  paste0(round(wk5$fav_off_epa_play_roll5,4)," (OK)"))))

cat("\n=== 6. FLIPPY EXCLUSION ===\n")
flippers_in<-intersect(ff,sf)
cat(sprintf("  Flippers in model: %d %s\n",length(flippers_in),if(length(flippers_in)==0)"GOOD"else"BUG"))

cat("\n=== 7. COMPOSITES ===\n")
for(f in c("nocover_conviction","spread_overextension","variance_env","dog_momentum_edge")){
  if(!f%in%names(v3)){cat(sprintf("  %s: MISSING\n",f));next}
  x<-v3[[f]];v<-!is.na(x)
  if(sum(v)<100){cat(sprintf("  %s: only %d valid\n",f,sum(v)));next}
  cat(sprintf("  %s: cor=%+.4f spread_cor=%+.4f n=%d\n",f,
    cor(x[v],nc_target[v]),cor(x[v],v3$spread_fav[v]),sum(v)))
}

cat("\n=== 8. SAMPLE SIZES ===\n")
for(ts in c(2023,2024,2025)){
  for(s in c(2012,2016,2020,2022)){
    n<-sum(v3$season>=s&v3$season<ts)
    cat(sprintf("  %d train%d+: %d games\n",ts,s,n))
  }
}

cat("\nAudit complete.\n")
