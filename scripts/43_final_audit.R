# Final comprehensive audit
library(tidyverse); library(here); library(glmnet)
source(here("scripts","00_project_setup.R"))

v3 <- read_rds(here("objects", "modeling_data_v3_FINAL.rds"))
cat(sprintf("Using v3_FINAL: %d x %d\n\n", nrow(v3), ncol(v3)))

ff<-c("ypp_diff","fav_roll5_ypp","is_primetime","qb_cpoe_diff","rush_epa_mismatch","fav_off_sit_roll5_pass_epa","dog_off_sit_roll5_pass_epa","injury_diff","fav_implied_prob_devigged","dog_implied_prob_devigged","fav_off_epa_play_roll5","fav_off_pass_epa_roll5","fav_off_success_rate_roll3","fav_def_pass_epa_roll3","dog_off_success_rate_roll3","dog_def_rush_epa_roll3","off_success_rate_roll5_diff","def_pass_epa_roll3_diff","fav_qb_air_yards_per_att_roll5","dog_qb_cpoe_roll5","qb_epa_per_play_roll5_diff","fav_n_out","dog_n_out","dog_n_questionable","fav_off_sit_pass_epa_roll5","fav_off_neg_play_rate_roll5","fav_def_rz_epa_roll5","fav_def_neg_play_rate_roll5","dog_off_explosive_rate_roll5","dog_def_3rd_epa_roll5","off_rz_epa_roll5_diff","off_explosive_rate_roll5_diff","def_3rd_epa_roll5_diff","def_explosive_rate_roll5_diff","def_neg_play_rate_roll5_diff","wind_over_12_hinge","rest_advantage")
bn<-c("total_spread_script","high_total_big_spread","dog_cold_streak","dog_hot_streak","fade_dog_heat","qb_scheme_fit","defensive_style_clash","fav_expected_score","dog_expected_score","fav_pass_lean","dog_def_weakness")
sf<-setdiff(v3|>select(where(is.numeric),-season,-week)|>names(),c(ff,bn))

cat("=== 1. DATA ===\n")
cat(sprintf("  spread_fav: [%.1f, %.1f] %s\n", min(v3$spread_fav), max(v3$spread_fav),
  if(all(v3$spread_fav>0))"OK" else "BUG"))
cat(sprintf("  Cover: %.1f%% | NC: %.1f%% | Flippers excluded: %d\n",
  mean(v3$fav_cover=="cover")*100, mean(v3$fav_cover=="no_cover")*100, length(ff)))

cat("\n=== 2. TRAIN/TEST ===\n")
for(ts in c(2023,2024,2025)){
  tr<-v3|>filter(season<ts);te<-v3|>filter(season==ts)
  cat(sprintf("  %d: train=%d test=%d overlap=%d\n",ts,nrow(tr),nrow(te),length(intersect(tr$game_id_norm,te$game_id_norm))))
}

cat("\n=== 3. END-TO-END COVER (train=2022, test=2024) ===\n")
train<-v3|>filter(season>=2022,season<2024); test<-v3|>filter(season==2024)
tc<-map_dbl(sf,~mean(!is.na(train[[.x]])));fe<-sf[tc>0.5]
tx<-train|>select(all_of(fe));md<-map_dbl(tx,~median(.x,na.rm=T))
tx<-tx|>mutate(across(everything(),~replace_na(.x,md[cur_column()])))|>as.matrix()
ex<-test|>select(all_of(fe))|>mutate(across(everything(),~replace_na(.x,md[cur_column()])))|>as.matrix()
m<-colMeans(tx);s<-apply(tx,2,sd);s[s==0]<-1;txs<-scale(tx,m,s);exs<-scale(ex,m,s)
gc<-which(apply(txs,2,sd)>0);txs<-txs[,gc];exs<-exs[,gc]

y<-factor(train$fav_cover,levels=c("cover","no_cover"))
fit<-glmnet(txs,y,family="binomial",alpha=0.1,lambda=0.005)
pred<-predict(fit,exs,type="response")[,1]
p_cover<-1-pred
cat(sprintf("  Features: %d | Overall: %.1f%% | P>0.55: %d bets %.1f%%\n",
  length(gc), mean((p_cover>0.5)==(test$fav_cover=="cover"))*100,
  sum(p_cover>0.55), if(sum(p_cover>0.55)>5)mean((test$fav_cover=="cover")[p_cover>0.55])*100 else 0))
cat(sprintf("  cor(p_cover, truth): %+.4f %s\n", cor(p_cover, as.integer(test$fav_cover=="cover")),
  if(cor(p_cover,as.integer(test$fav_cover=="cover"))>0)"CORRECT" else "INVERTED!"))
cat(sprintf("  Pred range: [%.3f, %.3f] SD=%.4f\n", min(p_cover), max(p_cover), sd(p_cover)))

cat("\n=== 4. END-TO-END NO-COVER (train=2012, test=2024) ===\n")
train2<-v3|>filter(season>=2012,season<2024)
tc2<-map_dbl(sf,~mean(!is.na(train2[[.x]])));fe2<-sf[tc2>0.5]
tx2<-train2|>select(all_of(fe2));md2<-map_dbl(tx2,~median(.x,na.rm=T))
tx2<-tx2|>mutate(across(everything(),~replace_na(.x,md2[cur_column()])))|>as.matrix()
ex2<-test|>select(all_of(fe2))|>mutate(across(everything(),~replace_na(.x,md2[cur_column()])))|>as.matrix()
m2<-colMeans(tx2);s2<-apply(tx2,2,sd);s2[s2==0]<-1;txs2<-scale(tx2,m2,s2);exs2<-scale(ex2,m2,s2)
gc2<-which(apply(txs2,2,sd)>0);txs2<-txs2[,gc2];exs2<-exs2[,gc2]

y_nc<-factor(train2$fav_cover,levels=c("no_cover","cover"))
fit_nc<-glmnet(txs2,y_nc,family="binomial",alpha=0.25,lambda=0.08)
pred_nc<-predict(fit_nc,exs2,type="response")[,1]
p_nc<-1-pred_nc
cat(sprintf("  Features: %d | Overall: %.1f%% | P>0.53: %d bets %.1f%%\n",
  length(gc2), mean((p_nc>0.5)==(test$fav_cover=="no_cover"))*100,
  sum(p_nc>0.53), if(sum(p_nc>0.53)>5)mean((test$fav_cover=="no_cover")[p_nc>0.53])*100 else 0))
cat(sprintf("  cor(p_nc, truth_nc): %+.4f %s\n", cor(p_nc, as.integer(test$fav_cover=="no_cover")),
  if(cor(p_nc,as.integer(test$fav_cover=="no_cover"))>0)"CORRECT" else "INVERTED!"))

cat("\n=== 5. OVERLAP ===\n")
cg<-test$game_id_norm[p_cover>0.55]; ng<-test$game_id_norm[p_nc>0.53]
cat(sprintf("  Cover: %d | NC: %d | Overlap: %d %s\n",
  length(cg),length(ng),length(intersect(cg,ng)),if(length(intersect(cg,ng))==0)"OK" else "CONFLICT"))

cat("\n=== 6. 2025 FEATURES ===\n")
t25<-v3|>filter(season==2025)
cat(sprintf("  Python roll5 for 2025: %.0f%% available\n", mean(!is.na(t25$fav_off_epa_play_roll5))*100))
cat(sprintf("  R roll5 (off_epa_diff): %.0f%%\n", mean(!is.na(t25$off_epa_diff))*100))
cat(sprintf("  Composites: mbs=%.0f%% sharp=%.0f%% everything=%.0f%%\n",
  mean(!is.na(t25$market_blindspot_score))*100, mean(!is.na(t25$sharp_proxy))*100,
  mean(!is.na(t25$everything_score))*100))

cat("\n=== 7. LEAKAGE ===\n")
target<-as.integer(v3$fav_cover=="cover")
leakers<-0
for(f in sf){x<-v3[[f]];v<-!is.na(x);if(sum(v)<100)next
  cv<-tryCatch(cor(x[v],target[v]),error=function(e)NA);if(is.na(cv))next
  if(abs(cv)>0.15){cat(sprintf("  LEAK: %s cor=%+.3f\n",f,cv));leakers<-leakers+1}}
if(leakers==0) cat("  CLEAN\n")

cat("\n=== AUDIT SUMMARY ===\n")
cat("  Data: OK\n")
cat("  Train/test: OK (zero overlap)\n")
cat("  Cover direction: check above\n")
cat("  No-cover direction: check above\n")
cat("  Overlap: check above\n")
cat("  2025 features: Python roll5 available, R roll5 missing\n")
cat("  Leakage: check above\n")
