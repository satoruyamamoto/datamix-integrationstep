# pythonでクラスタリングをして、クラスターごとに分けたデータを読み込んで、予測モデルを作成する。
# Rでの生存時間解析（時間変化共変量の処理）
# eventを予測する
# データの読み込み 初期のデータ
# 必要ない列(Nan)は削除済み
train_data_cluster0<-read.csv('train_data/train_data_cluster0.csv')
train_data_cluster1<-read.csv('train_data/train_data_cluster1.csv')

test_data_cluster0<-read.csv('test_data/test_data_cluster0.csv')
test_data_cluster1<-read.csv('test_data/test_data_cluster1.csv')

## event列を追加　RUL<=100ならば1,100<RULならば0とする
train_data_cluster0$event<-ifelse(train_data_cluster0$RUL <= 100, 1,0)
train_data_cluster1$event<-ifelse(train_data_cluster1$RUL <= 100, 1,0)

# 終わりの時間stopを追加time_in_cyclesが1刻みなのでtime_in_cycles+1とする
train_data_cluster0$stop <- train_data_cluster0$time_in_cycles+1
train_data_cluster1$stop <- train_data_cluster1$time_in_cycles+1
test_data_cluster0$stop <- test_data_cluster0$time_in_cycles+1
test_data_cluster1$stop <- test_data_cluster1$time_in_cycles+1

# 予測モデルを作成する
library(survival)


# cluster0
Survival_event_time_cluster0 <- coxph(data=train_data_cluster0,
                        Surv(time_in_cycles,stop,event)~op_setting_1 + op_setting_2 + op_setting_3 + sensor_1
                        + sensor_2 + sensor_3 + sensor_4 + sensor_5 + sensor_6 + sensor_7 + sensor_8
                        + sensor_9 + sensor_10 + sensor_11 + sensor_12 + sensor_13 + sensor_14 + sensor_15
                        + sensor_16 + sensor_17 + sensor_18 + sensor_19     
                        + sensor_20 + sensor_21)
summary(Survival_event_time_cluster0)

# step関数で説明変数を選択する
step(Survival_event_time_cluster0)

# 再度モデル定義
Survival_event_time_cluster0_step <- coxph(data=train_data_cluster0,
                                           Surv(time_in_cycles, stop, event) ~ op_setting_1 + 
                                             op_setting_3 + sensor_1 + sensor_2 + sensor_3 + sensor_4 + 
                                             sensor_6 + sensor_9 + sensor_10 + sensor_11 + sensor_12 + 
                                             sensor_14 + sensor_16 + sensor_17 + sensor_18 + sensor_20 + 
                                             sensor_21)


# test_dataで予測する
pred_test_time_cluster0_step = predict(Survival_event_time_cluster0_step, 
                                  newdata = test_data_cluster0, 
                                  type=c("lp"))

#上記の予測結果の列を追加する
test_data_cluster0$pred_test_time_step <- pred_test_time_cluster0_step

# eventが1になる確率に変換して、列を追加する
pred_test2_time_cluster0_step <-1 / (1 + exp(-pred_test_time_cluster0_step))
test_data_cluster0$pred_test2_time_step <- pred_test2_time_cluster0_step


# 上記の予測結果は閾値によって変わるので、複数の閾値を検証する。
# 閾値を0.5に設定
pred_test2_time_flag_cluster0_step<-ifelse(pred_test2_time_cluster0_step > 0.5, 1, 0)
test_data_cluster0$pred_test2_time_flag_step<-pred_test2_time_flag_cluster0_step
hist(pred_test2_time_flag_cluster0_step)

# csvで出力して、pythonで混合行列を確認する。
write.csv(test_data_cluster0, "test_data/test_data_R_time_cluster0_0.5.csv")


# 閾値を0.4に設定
pred_test2_time_flag_cluster0_step<-ifelse(pred_test2_time_cluster0_step > 0.4, 1, 0)
test_data_cluster0$pred_test2_time_flag_step<-pred_test2_time_flag_cluster0_step
hist(pred_test2_time_flag_cluster0_step)

# csvで出力して、pythonで混合行列を確認する。
write.csv(test_data_cluster0, "test_data/test_data_R_time_cluster0_0.4.csv")

# 閾値を0.6に設定
pred_test2_time_flag_cluster0_step<-ifelse(pred_test2_time_cluster0_step > 0.6, 1, 0)
test_data_cluster0$pred_test2_time_flag_step<-pred_test2_time_flag_cluster0_step
hist(pred_test2_time_flag_cluster0_step)

# csvで出力して、pythonで混合行列を確認する。
write.csv(test_data_cluster0, "test_data/test_data_R_time_cluster0_0.6.csv")



# cluster1
Survival_event_time_cluster1 <- coxph(data=train_data_cluster1,
                                      Surv(time_in_cycles,stop,event)~op_setting_1 + op_setting_2 + op_setting_3 + sensor_1
                                      + sensor_2 + sensor_3 + sensor_4 + sensor_5 + sensor_6 + sensor_7 + sensor_8
                                      + sensor_9 + sensor_10 + sensor_11 + sensor_12 + sensor_13 + sensor_14 + sensor_15
                                      + sensor_16 + sensor_17 + sensor_18 + sensor_19     
                                      + sensor_20 + sensor_21)
summary(Survival_event_time_cluster1)

# step関数で説明変数を選択する
step(Survival_event_time_cluster1)

# 再度モデル定義
Survival_event_time_cluster1_step <- coxph(data=train_data_cluster1,
                                           Surv(time_in_cycles, stop, event) ~ sensor_2 + 
                                             sensor_3 + sensor_4 + sensor_6 + sensor_8 + sensor_9 + sensor_10 + 
                                             sensor_11 + sensor_13 + sensor_15 + sensor_17 + sensor_20 + 
                                             sensor_21)


# test_dataで予測する
pred_test_time_cluster1_step = predict(Survival_event_time_cluster1_step, 
                                       newdata = test_data_cluster1, 
                                       type=c("lp"))

#上記の予測結果の列を追加する
test_data_cluster1$pred_test_time_step <- pred_test_time_cluster1_step

# eventが1になる確率に変換して、列を追加する
pred_test2_time_cluster1_step <-1 / (1 + exp(-pred_test_time_cluster1_step))
test_data_cluster1$pred_test2_time_step <- pred_test2_time_cluster1_step


# 上記の予測結果は閾値によって変わるので、複数の閾値を検証する。
# 閾値を0.5に設定
pred_test2_time_flag_cluster1_step<-ifelse(pred_test2_time_cluster1_step > 0.5, 1, 0)
test_data_cluster1$pred_test2_time_flag_step<-pred_test2_time_flag_cluster1_step
hist(pred_test2_time_flag_cluster1_step)

# csvで出力して、pythonで混合行列を確認する。
write.csv(test_data_cluster1, "test_data/test_data_R_time_cluster1_0.5.csv")


# 閾値を0.4に設定
pred_test2_time_flag_cluster1_step<-ifelse(pred_test2_time_cluster1_step > 0.4, 1, 0)
test_data_cluster1$pred_test2_time_flag_step<-pred_test2_time_flag_cluster1_step
hist(pred_test2_time_flag_cluster1_step)

# csvで出力して、pythonで混合行列を確認する。
write.csv(test_data_cluster1, "test_data/test_data_R_time_cluster1_0.4.csv")

# 閾値を0.6に設定
pred_test2_time_flag_cluster1_step<-ifelse(pred_test2_time_cluster1_step > 0.6, 1, 0)
test_data_cluster1$pred_test2_time_flag_step<-pred_test2_time_flag_cluster1_step
hist(pred_test2_time_flag_cluster1_step)

# csvで出力して、pythonで混合行列を確認する。
write.csv(test_data_cluster1, "test_data/test_data_R_time_cluster1_0.6.csv")


