#!/usr/bin/env python
# coding: utf-8

# # 予測モデルを作成する。予測モデルは全データ一括のパターンとclusterごとに予測するパターンを作成し、比較する。

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, f1_score,classification_report
from random import randint, shuffle
import os
import seaborn as sns



# In[ ]:


train_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/train_data/train_data.csv')


# In[ ]:


test_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/test_data/test_data.csv')


# # 全データ一括 

# In[18]:


# 欠損値のある列とengine_no,time_in_cycles以外で特徴量選択
selected_features_train_model = [x for x in train_data.columns if x not in nan_column]


# In[19]:


selected_features_train_model.remove('engine_no')
selected_features_train_model.remove('time_in_cycles')
selected_features_train_model.remove('RUL')


# In[20]:


# 大元のtrain_dataをx(selected_features)とy(RUL)に分ける
X_train, y_train = train_data[selected_features_train_model], train_data['RUL']


# In[21]:


X_train.head()


# In[22]:


# 欠損値のある列とengine_no,time_in_cycles以外で特徴量選択
selected_features_test_model = [x for x in test_data.columns if x not in nan_column]


# In[23]:


selected_features_test_model.remove('engine_no')
selected_features_test_model.remove('time_in_cycles')


# In[24]:


# 大元のtestの特徴量からselected_featuresだけのX_testにする
X_test = test_data[selected_features_test_model]


# In[25]:


X_test.head()


# In[103]:


# ランダムフォレスト
rf = RandomForestRegressor(random_state=1234)

# train_dataで学習
rf.fit(X_train, y_train)

# FutureWarningは問題ありません


# In[104]:


# 予測する
train_data['pred_rf'] = rf.predict(X_train)
test_data['pred_rf'] = rf.predict(X_test)


# In[105]:


train_data.head()


# In[106]:


test_data.head()


# In[110]:


# 結果表の作成 
train_data = train_data.sort_values(['engine_no', 'time_in_cycles'])
test_data = test_data.sort_values(['engine_no', 'time_in_cycles'])

# test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf'のみ
result = test_data.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf']]


# In[111]:


result.head()


# In[113]:


# pred_rf<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result['result'] = result['pred_rf'].map(lambda x: 0 if x > 100 else 1)


# In[114]:


result.head()


# In[115]:


# 結果の保存
result[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/submission.csv', index=False)


# In[17]:


# testデータのRULを読み込む
test_data_RUL = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/RUL_test.csv')


# In[18]:


# 列の順番を入れ替えと削除
test_data_RUL.drop('Unnamed: 0', axis=1)
test_data_RUL = test_data_RUL.loc[:,['engine_no','RUL']]
test_data_RUL.head()


# In[19]:


# RUL<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
test_data_RUL['result'] = test_data_RUL['RUL'].map(lambda x: 0 if x > 100 else 1)


# In[20]:


test_data_RUL.head()


# In[123]:


# 混合行列まとめて表示
cr = classification_report(test_data_RUL['result'],result['result'] )
print(cr)


# In[126]:


# 特徴量の重要度を算出
feature_importances = rf.feature_importances_


feature_importance =     pd.DataFrame(rf.feature_importances_, columns=["importance"], index=selected_features_test_model)
feature_importance.sort_values("importance", ascending=False)


# In[127]:


feature_importance.sort_values("importance", ascending=False).plot(kind="bar")


# ##特徴量は\
# sensor_10	0.000794\
# sensor_16	0.000233\
# sensor_1	0.000020\
# sensor_18	0.000009\
# sensor_5	0.000004\
# op_setting_3	0.000000\
# sensor_19	0.000000\
# はほとんど意味がないと推測できる

# # ランダムフォレスト　最初の欠損値に加え、上記の特徴量を削る

# In[26]:


selected_features_test_model


# In[27]:


serected_features_importance = selected_features_test_model


# In[28]:


serected_features_importance.remove('sensor_10')


# In[29]:


serected_features_importance.remove('sensor_16')


# In[30]:


serected_features_importance.remove('sensor_1')


# In[31]:


serected_features_importance.remove('sensor_18')


# In[32]:


serected_features_importance.remove('sensor_5')


# In[33]:


serected_features_importance.remove('op_setting_3')


# In[34]:


serected_features_importance.remove('sensor_19')


# In[35]:


serected_features_importance


# In[36]:


#大元のtrainをx(selected_features)とy(RUL)に分ける
X_train_all_rf_fe_im, y_train_all_rf_fe_im = train_data[serected_features_importance], train_data['RUL']


# In[37]:


#大元のtestの特徴量からselected_featuresだけのX_test_rf_fe_imにする
X_test_rf_fe_im = test_data[serected_features_importance]


# In[142]:


#ランダムフォレスト
rf_fe_im = RandomForestRegressor(random_state=1234)
rf_fe_im.fit(X_train_all_rf_fe_im, y_train_all_rf_fe_im)


# FutureWarningは問題ありません


# In[143]:


#予測する
train_data['pred_rf_fe_im'] = rf_fe_im.predict(X_train_all_rf_fe_im)
test_data['pred_rf_fe_im'] = rf_fe_im.predict(X_test_rf_fe_im)


# In[144]:


train_data.head()


# In[145]:


test_data.head()


# In[147]:


#結果表の作成 
train_data = train_data.sort_values(['engine_no', 'time_in_cycles'])
test_data = test_data.sort_values(['engine_no', 'time_in_cycles'])

# #test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf'のみ
result_rf_fe_im = test_data.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_fe_im']]


# In[148]:


result_rf_fe_im.head()


# In[149]:


# pred_rf<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_rf_fe_im['result'] = result_rf_fe_im['pred_rf_fe_im'].map(lambda x: 0 if x > 100 else 1)


# In[150]:


result_rf_fe_im.head()


# In[151]:


#結果の保存
result_rf_fe_im[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/submission2.csv', index=False)


# In[152]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL['result'],result_rf_fe_im['result'] )
print(cr)


# # クロスバリデーションによるハイパーパラメータのチューニング 上記のfeature importanceで行う

# In[38]:


#クロスバリデーションによるハイパーパラメータのチューニング
rf_gscv_fe_im = RandomForestRegressor(random_state=1234)


# In[154]:


params = {'n_estimators': [200, 500, 1000], 'max_depth': [20, 50, 500]}


# In[155]:


## 注意: scikit-learnのバージョンが、0.18以降の場合は、scoring='neg_mean_squared_error'とします
rf_gscv_fe_im = GridSearchCV(rf_gscv_fe_im, param_grid=params, verbose=1,
                    cv=3, scoring='neg_mean_squared_error')

# #注意: scikit-learnのバージョンが、0.17の場合は、scoring='mean_squared_error'とします
# import sklearn
# sklearn.__version__
# gscv = GridSearchCV(rf, param_grid=params, verbose=1,
#                     cv=3, scoring='mean_squared_error')


# In[ ]:


rf_gscv_fe_im.fit(X_train_all_rf_fe_im, y_train_all_rf_fe_im)


# In[101]:


rf_gscv_fe_im.best_params_


# In[39]:


# ハイパーパラメータチューニング後で学習する
rf_gscv_fe_im_tu_all = RandomForestRegressor(n_estimators=200, max_depth=50, random_state=1234)
rf_gscv_fe_im_tu_all.fit(X_train_all_rf_fe_im, y_train_all_rf_fe_im)


# In[46]:


#上記のrf_gscv_fe_im_tu_allで学習したモデルでtest_fe_imデータを予測する
train_data['pred_rf_gscv_fe_im_tu_all'] = rf_gscv_fe_im_tu_all.predict(X_train_all_rf_fe_im)
test_data['pred_rf_gscv_fe_im_tu_all'] = rf_gscv_fe_im_tu_all.predict(X_test_rf_fe_im)


# In[47]:


train_data.head()


# In[48]:


test_data.head()


# In[49]:


#結果表の作成 
train_data = train_data.sort_values(['engine_no', 'time_in_cycles'])
test_data = test_data.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf_gscv_all'のみ
result_rf_gscv_fe_im_tu = test_data.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_gscv_fe_im_tu_all']]


# In[50]:


result_rf_gscv_fe_im_tu.head()


# In[51]:


# pred_rf_gscv_all<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_rf_gscv_fe_im_tu['result'] = result_rf_gscv_fe_im_tu['pred_rf_gscv_fe_im_tu_all'].map(lambda x: 0 if x > 100 else 1)


# In[52]:


result_rf_gscv_fe_im_tu.head()


# In[53]:


#結果の保存
result_rf_gscv_fe_im_tu[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/submission3.csv', index=False)


# In[54]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL['result'],result_rf_gscv_fe_im_tu['result'] )
print(cr)


# # STEP1で作成したcluster0のtrain_dataとtest_dataを読み込む

# In[81]:


# cluster0のtrain_dataとtest_dataを読み込み
train_data_cluster0 = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster0.csv')
test_data_cluster0 = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster0.csv')


# # STEP1で作成したcluster0test_dataのRUL（答え）をclusterごとに分けて、他必要な処理を行ったファイルを読み込む

# In[82]:


test_data_RUL_cluster0 = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_RUL_cluster0.csv')


# # cluster0

# In[83]:


#重複した行を削除
list_engine_no_train_data_cluster0 = list(train_data_cluster0['engine_no'].drop_duplicates())


# In[84]:


list_engine_no_train_data_cluster0


# In[85]:


# 特徴量を抽出　上記の欠損値以外を抽出
metadata_columns = ['engine_no', 'time_in_cycles']
selected_features = [x for x in test_data_cluster0.columns if x not in metadata_columns]


# #  特徴量選択をやり直す際はここから行う

# In[86]:


selected_features


# #  ここで特徴量を入れていくので、特徴量選択をし直した後はここからやり直す

# In[87]:


#大元のtrainをx(selected_features)とy(RUL)に分ける
X_train_all_cluster0, y_train_all_cluster0 = train_data_cluster0[selected_features], train_data_cluster0['RUL']


# In[88]:


#大元のtestの特徴量からselected_featuresだけのX_testにする
X_test_cluster0 = test_data_cluster0[selected_features]


# ## ランダムフォレスト　欠損値の列のみ除く

# In[89]:


#ランダムフォレスト
rf_cluster0 = RandomForestRegressor(random_state=1234)
rf_cluster0.fit(X_train_all_cluster0, y_train_all_cluster0)


# FutureWarningは問題ありません


# # 注意　最後はcluster1と結合するので、列名はpred_rfで統一する

# In[90]:


# 予測する
test_data_cluster0['pred_rf'] = rf_cluster0.predict(X_test_cluster0)


# In[91]:


test_data_cluster0.head()


# # 注意　最後は結合するので、列名はpred_rfで統一する

# In[92]:


#結果表の作成 
train_data_cluster0 = train_data_cluster0.sort_values(['engine_no', 'time_in_cycles'])
test_data_cluster0 = test_data_cluster0.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf'のみ
result_cluster0 = test_data_cluster0.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf']]


# In[93]:


result_cluster0.head()


# In[94]:


# pred_rf<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_cluster0['result'] = result_cluster0['pred_rf'].map(lambda x: 0 if x > 100 else 1)

#lambda:無名関数（匿名関数）は文字通り名前のない関数で、一度きりの使い捨ての関数として使います。無名関数を使うと処理を簡潔に書けたり、グローバル・スコープの関数（オブジェクト）を生成せずに済むといったメリットがあります。
#組み込み関数map()と無名関数lambda（ラムダ式）とをあわせて使うと、リストの要素すべてを２倍する、（文字列要素を）一括置換する、といったようにリスト要素すべてに変更を加えたオブジェクトを取得することができます。


# In[95]:


result_cluster0.head()


# In[96]:


#csvで結果の保存
result_cluster0[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/cluster0_submission1.csv', index=False)


# # 上記で読み込んだtest_data_RUL_cluster0を使っていく

# In[97]:


test_data_RUL_cluster0.head()


# # train_allで学習しtestデータで出した予測値と、実際の答えで混合行列を作成

# In[98]:


print((result_cluster0['result'] == 1).sum())
print((result_cluster0['result'] == 0).sum())


# In[99]:


print((test_data_RUL_cluster0['result'] == 1).sum())
print((test_data_RUL_cluster0['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[100]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL_cluster0['result'],result_cluster0['result'],labels=[1, 0] )


# In[101]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)233、真陰性: TN (True-Negative)177、偽陽性: FP (False-Positive)15(故障していないのにしていると予測)、偽陰性: FN (False-Negative)82(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)82を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[102]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL_cluster0['result'],result_cluster0['result'],labels=[1, 0] )
print(cr)


# # labelsを付けた時の計算が上記のまとめて表示の1の方　labelsをつけなかったときの計算が下の方

# # #再現率を上げたい　実際の故障のうち故障と予測できたもの

# In[103]:


#特徴量の重要度を算出
feature_importances_cluster0 = rf_cluster0.feature_importances_


feature_importance_cluster0 =     pd.DataFrame(rf_cluster0.feature_importances_, columns=["importance"], index=selected_features)
feature_importance_cluster0.sort_values("importance", ascending=False)

#全部足すと100% 縦軸で
#これでは重要度はわかるが、回帰のように上がる下がるなど関係性はわからない


# In[104]:


feature_importance_cluster0.sort_values("importance", ascending=False).plot(kind="bar")


# ##特徴量は\
# sensor_10	1.049026e-03\
# sensor_16	2.710828e-04\
# sensor_1	1.296134e-05\
# sensor_18	6.439914e-06\
# sensor_5	4.552048e-06\
# sensor_19	2.486474e-07\
# op_setting_3	1.300208e-08\
# はほとんど意味がないと推測できる

# # ランダムフォレスト　最初の欠損値に加え、上記の特徴量を削る

# In[105]:


selected_features


# In[106]:


serected_features_importance_cluster0 = selected_features
serected_features_importance_cluster0.remove('sensor_10')


# In[107]:


serected_features_importance_cluster0.remove('sensor_16')


# In[108]:


serected_features_importance_cluster0.remove('sensor_1')


# In[109]:


serected_features_importance_cluster0.remove('sensor_18')


# In[110]:


serected_features_importance_cluster0.remove('sensor_5')


# In[111]:


serected_features_importance_cluster0.remove('sensor_19')


# In[112]:


serected_features_importance_cluster0.remove('op_setting_3')


# In[113]:


serected_features_importance_cluster0


# In[114]:


#大元のtrainをx(selected_features)とy(RUL)に分ける
X_train_all_cluster0_rf_fe_im, y_train_all_cluster0_rf_fe_im = train_data_cluster0[serected_features_importance_cluster0], train_data_cluster0['RUL']


# In[115]:


#大元のtestの特徴量からselected_featuresだけのX_test_rf_fe_imにする
X_test_cluster0_rf_fe_im = test_data_cluster0[serected_features_importance_cluster0]


# In[116]:


#ランダムフォレスト
rf_cluster0_fe_im = RandomForestRegressor(random_state=1234)
rf_cluster0_fe_im.fit(X_train_all_cluster0_rf_fe_im, y_train_all_cluster0_rf_fe_im)


# FutureWarningは問題ありません


# # 注意　最後は結合するので、列名はpred_rf_fe_imで統一する 
# 

# In[117]:


#予測する
train_data_cluster0['pred_rf_fe_im'] = rf_cluster0_fe_im.predict(X_train_all_cluster0_rf_fe_im)
test_data_cluster0['pred_rf_fe_im'] = rf_cluster0_fe_im.predict(X_test_cluster0_rf_fe_im)


# In[118]:


train_data_cluster0.head()


# In[119]:


test_data_cluster0.head()


# In[120]:


#結果表の作成 
train_data_cluster0 = train_data_cluster0.sort_values(['engine_no', 'time_in_cycles'])
test_data_cluster0 = test_data_cluster0.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf'のみ
result_cluster0_rf_fe_im = test_data_cluster0.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_fe_im']]


# In[121]:


result_cluster0_rf_fe_im.head()


# In[122]:


# pred_rf<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_cluster0_rf_fe_im['result'] = result_cluster0_rf_fe_im['pred_rf_fe_im'].map(lambda x: 0 if x > 100 else 1)


# In[123]:


result_cluster0_rf_fe_im.head()
# result_cluster0_rf_fe_im = result_cluster0_rf_fe_im.drop("resul_cluster0t", axis=1)


# In[124]:


#csvで結果の保存
result_cluster0_rf_fe_im[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/cluster0_submission2.csv', index=False)


# # 上記で読み込んだtest_data_RUL_cluster0を使っていく

# In[125]:


test_data_RUL_cluster0.head()


# # train_allで学習しtestデータで出した予測値と、実際の答えで混合行列を作成

# In[126]:


print((result_cluster0_rf_fe_im['result'] == 1).sum())
print((result_cluster0_rf_fe_im['result'] == 0).sum())


# In[127]:


print((test_data_RUL_cluster0['result'] == 1).sum())
print((test_data_RUL_cluster0['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[128]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL_cluster0['result'],result_cluster0_rf_fe_im['result'],labels=[1, 0] )


# In[129]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)233、真陰性: TN (True-Negative)177、偽陽性: FP (False-Positive)15(故障していないのにしていると予測)、偽陰性: FN (False-Negative)82(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)82を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[130]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL_cluster0['result'],result_cluster0_rf_fe_im['result'],labels=[1, 0] )
print(cr)


# # #再現率を上げたい　実際の故障のうち故障と予測できたもの

# In[131]:


#特徴量の重要度を算出
feature_importances_cluster0_rf_fe_im = rf_cluster0_fe_im.feature_importances_


feature_importance_cluster0_rf_fe_im =     pd.DataFrame(rf_cluster0_fe_im.feature_importances_, columns=["importance"], index=selected_features)
feature_importance_cluster0_rf_fe_im.sort_values("importance", ascending=False)

#全部足すと100% 縦軸で
#これでは重要度はわかるが、回帰のように上がる下がるなど関係性はわからない


# In[132]:


feature_importance_cluster0_rf_fe_im.sort_values("importance", ascending=False).plot(kind="bar")


# In[133]:


#メモリを軽くする
import gc
 
del result_cluster0
del rf_cluster0
del rf_cluster0_fe_im
gc.collect()


# # クロスバリデーションによるハイパーパラメータのチューニング 上記のfeature importanceで行う

# In[134]:


#クロスバリデーションによるハイパーパラメータのチューニング
rf_cluster0_gscv_fe_im = RandomForestRegressor(random_state=1234)


# In[135]:


params = {'n_estimators': [100, 150, 200], 'max_depth': [50, 100, 150]}


# In[136]:


## 注意: scikit-learnのバージョンが、0.18以降の場合は、scoring='neg_mean_squared_error'とします
rf_cluster0_gscv_fe_im = GridSearchCV(rf_cluster0_gscv_fe_im, param_grid=params, verbose=1,
                    cv=3, scoring='neg_mean_squared_error')

# #注意: scikit-learnのバージョンが、0.17の場合は、scoring='mean_squared_error'とします
# import sklearn
# sklearn.__version__
# gscv = GridSearchCV(rf, param_grid=params, verbose=1,
#                     cv=3, scoring='mean_squared_error')


# In[137]:


rf_cluster0_gscv_fe_im.fit(X_train_all_cluster0_rf_fe_im, y_train_all_cluster0_rf_fe_im)


# In[138]:


rf_cluster0_gscv_fe_im.best_params_


# In[139]:


#train_all_fe_imデータで学習しなおす
rf_cluster0_gscv_fe_im_tu_all = RandomForestRegressor(n_estimators=200, max_depth=50, random_state=1234)
rf_cluster0_gscv_fe_im_tu_all.fit(X_train_all_cluster0_rf_fe_im, y_train_all_cluster0_rf_fe_im)


# In[140]:


#上記のrf_gscv_fe_im_tu_allで学習したモデルでtest_fe_imデータを予測する
train_data_cluster0['pred_rf_gscv_fe_im_tu_all'] = rf_cluster0_gscv_fe_im_tu_all.predict(X_train_all_cluster0_rf_fe_im)
test_data_cluster0['pred_rf_gscv_fe_im_tu_all'] = rf_cluster0_gscv_fe_im_tu_all.predict(X_test_cluster0_rf_fe_im)


# In[141]:


train_data_cluster0.head()


# In[142]:


test_data_cluster0.head()


# In[143]:


#結果表の作成 
train_data_cluster0 = train_data_cluster0.sort_values(['engine_no', 'time_in_cycles'])
test_data_cluster0 = test_data_cluster0.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf_gscv_all'のみ
result_cluster0_rf_gscv_fe_im_tu = test_data_cluster0.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_gscv_fe_im_tu_all']]


# In[144]:


result_cluster0_rf_gscv_fe_im_tu.head()


# In[145]:


# pred_rf_gscv_all<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_cluster0_rf_gscv_fe_im_tu['result'] = result_cluster0_rf_gscv_fe_im_tu['pred_rf_gscv_fe_im_tu_all'].map(lambda x: 0 if x > 100 else 1)


# In[146]:


result_cluster0_rf_gscv_fe_im_tu.head()


# In[147]:


#csvで結果の保存
result_cluster0_rf_gscv_fe_im_tu[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/cluster0_submission3.csv', index=False)


# # 上記で読み込んだtest_data_RUL_cluster0を使用する

# In[71]:


test_data_RUL_cluster0.head()


# # train_allで学習しtestデータで出した予測値と、実際の答えで混合行列を作成

# In[72]:


print((result_cluster0_rf_gscv_fe_im_tu['result'] == 1).sum())
print((result_cluster0_rf_gscv_fe_im_tu['result'] == 0).sum())


# In[73]:


print((test_data_RUL_cluster0['result'] == 1).sum())
print((test_data_RUL_cluster0['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[74]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL_cluster0['result'],result_cluster0_rf_gscv_fe_im_tu['result'],labels=[1, 0] )


# In[75]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)233、真陰性: TN (True-Negative)177、偽陽性: FP (False-Positive)15(故障していないのにしていると予測)、偽陰性: FN (False-Negative)82(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)82を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[76]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL_cluster0['result'],result_cluster0_rf_gscv_fe_im_tu['result'],labels=[1, 0] )
print(cr)


# # labelsを付けた時の計算が上記のまとめて表示の1の方　labelsをつけなかったときの計算が下の方

# # #再現率を上げたい　実際の故障のうち故障と予測できたもの

# In[77]:


#特徴量の重要度を算出
feature_importances_cluster0_rf_gscv_fe_im_tu = rf_cluster0_gscv_fe_im_tu_all.feature_importances_


feature_importances_cluster0_rf_gscv_fe_im_tu =     pd.DataFrame(rf_cluster0_gscv_fe_im_tu_all.feature_importances_, columns=["importance"], index=selected_features)
feature_importances_cluster0_rf_gscv_fe_im_tu.sort_values("importance", ascending=False)

#全部足すと100% 縦軸で
#これでは重要度はわかるが、回帰のように上がる下がるなど関係性はわからない


# In[78]:


feature_importances_cluster0_rf_gscv_fe_im_tu.sort_values("importance", ascending=False).plot(kind="bar")


# # 結果をtrain_data_cluster0_pred、test_data_cluster0_predとして出力する

# In[79]:


# cluster0のデータの予測結果追加を出力
train_data_cluster0.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster0_pred.csv', index=False)
test_data_cluster0.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster0_pred.csv', index=False)


# In[392]:


# cluster0の分けたデータの予測結果追加を読み込み
train_data_cluster0_pred = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster0_pred.csv')
test_data_cluster0_pred = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster0_pred.csv')


# # STEP1で作成したcluster1のtrain_dataとtest_dataを読み込む

# In[2]:


# cluster1のtrain_dataとtest_dataを読み込み
train_data_cluster1 = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster1.csv')
test_data_cluster1 = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster1.csv')


# # STEP1で作成したtest_dataのRUL（答え）をclusterごとに分けて、他必要な処理を行ったファイルを読み込む

# In[3]:


test_data_RUL_cluster1 = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_RUL_cluster1.csv')


# # cluster1

# In[4]:


#重複した行を削除
list_engine_no_train_data_cluster1 = list(train_data_cluster1['engine_no'].drop_duplicates())


# In[5]:


list_engine_no_train_data_cluster1


# In[6]:


# 特徴量を抽出　上記の欠損値以外を抽出
metadata_columns = ['engine_no', 'time_in_cycles']
selected_features = [x for x in test_data_cluster1.columns if x not in metadata_columns]


# #  特徴量選択をやり直す際はここから行う

# In[7]:


selected_features


# #  ここで特徴量を入れていくので、特徴量選択をし直した後はここからやり直す

# In[8]:


#大元のtrainをx(selected_features)とy(RUL)に分ける
X_train_all_cluster1, y_train_all_cluster1 = train_data_cluster1[selected_features], train_data_cluster1['RUL']


# In[9]:


#大元のtestの特徴量からselected_featuresだけのX_testにする
X_test_cluster1 = test_data_cluster1[selected_features]


# ## ランダムフォレスト　欠損値の列のみ除く

# In[10]:


#ランダムフォレスト
rf_cluster1 = RandomForestRegressor(random_state=1234)
rf_cluster1.fit(X_train_all_cluster1, y_train_all_cluster1)


# FutureWarningは問題ありません


# # 注意　最後は結合するので、列名はpred_rfで統一する

# In[11]:


#予測する
test_data_cluster1['pred_rf'] = rf_cluster1.predict(X_test_cluster1)


# In[12]:


test_data_cluster1.head()


# # 注意　最後は結合するので、列名はpred_rfで統一する

# In[13]:


#結果表の作成 
train_data_cluster1 = train_data_cluster1.sort_values(['engine_no', 'time_in_cycles'])
test_data_cluster1 = test_data_cluster1.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf'のみ
result_cluster1 = test_data_cluster1.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf']]


# In[14]:


result_cluster1.head()


# In[15]:


# pred_rf<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_cluster1['result'] = result_cluster1['pred_rf'].map(lambda x: 0 if x > 100 else 1)

#lambda:無名関数（匿名関数）は文字通り名前のない関数で、一度きりの使い捨ての関数として使います。無名関数を使うと処理を簡潔に書けたり、グローバル・スコープの関数（オブジェクト）を生成せずに済むといったメリットがあります。
#組み込み関数map()と無名関数lambda（ラムダ式）とをあわせて使うと、リストの要素すべてを２倍する、（文字列要素を）一括置換する、といったようにリスト要素すべてに変更を加えたオブジェクトを取得することができます。


# In[16]:


result_cluster1.head()


# In[17]:


#csvで結果の保存
result_cluster1[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/cluster1_submission1.csv', index=False)


# # 上記で読み込んだtest_data_RUL_cluster1を使っていく

# In[18]:


test_data_RUL_cluster1.head()


# # train_allで学習しtestデータで出した予測値と、実際の答えで混合行列を作成

# In[19]:


print((result_cluster1['result'] == 1).sum())
print((result_cluster1['result'] == 0).sum())


# In[20]:


print((test_data_RUL_cluster1['result'] == 1).sum())
print((test_data_RUL_cluster1['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[21]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL_cluster1['result'],result_cluster1['result'],labels=[1, 0] )


# In[22]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)233、真陰性: TN (True-Negative)177、偽陽性: FP (False-Positive)15(故障していないのにしていると予測)、偽陰性: FN (False-Negative)82(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)82を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[23]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL_cluster1['result'],result_cluster1['result'],labels=[1, 0] )
print(cr)


# # labelsを付けた時の計算が上記のまとめて表示の1の方　labelsをつけなかったときの計算が下の方

# # #再現率を上げたい　実際の故障のうち故障と予測できたもの

# In[24]:


#特徴量の重要度を算出
feature_importances_cluster1 = rf_cluster1.feature_importances_


feature_importance_cluster1 =     pd.DataFrame(rf_cluster1.feature_importances_, columns=["importance"], index=selected_features)
feature_importance_cluster1.sort_values("importance", ascending=False)

#全部足すと100% 縦軸で
#これでは重要度はわかるが、回帰のように上がる下がるなど関係性はわからない


# In[25]:


feature_importance_cluster1.sort_values("importance", ascending=False).plot(kind="bar")


# ##特徴量は\
# sensor_10	0.000116\
# sensor_5	0.000000\
# sensor_1	0.000000\
# op_setting_3	0.000000\
# sensor_16	0.000000\
# sensor_18	0.000000\
# sensor_19	0.000000\
# はほとんど意味がないと推測できる

# # ランダムフォレスト　最初の欠損値に加え、上記の特徴量を削る

# In[26]:


selected_features


# In[27]:


serected_features_importance_cluster1 = selected_features
serected_features_importance_cluster1.remove('sensor_10')


# In[28]:


serected_features_importance_cluster1.remove('sensor_5')


# In[29]:


serected_features_importance_cluster1.remove('sensor_1')


# In[30]:


serected_features_importance_cluster1.remove('op_setting_3')


# In[31]:


serected_features_importance_cluster1.remove('sensor_16')


# In[32]:


serected_features_importance_cluster1.remove('sensor_18')


# In[33]:


serected_features_importance_cluster1.remove('sensor_19')


# In[34]:


serected_features_importance_cluster1


# In[35]:


#大元のtrainをx(selected_features)とy(RUL)に分ける
X_train_all_cluster1_rf_fe_im, y_train_all_cluster1_rf_fe_im = train_data_cluster1[serected_features_importance_cluster1], train_data_cluster1['RUL']


# In[36]:


#大元のtestの特徴量からselected_featuresだけのX_test_rf_fe_imにする
X_test_cluster1_rf_fe_im = test_data_cluster1[serected_features_importance_cluster1]


# In[37]:


#ランダムフォレスト
rf_cluster1_fe_im = RandomForestRegressor(random_state=1234)
rf_cluster1_fe_im.fit(X_train_all_cluster1_rf_fe_im, y_train_all_cluster1_rf_fe_im)


# FutureWarningは問題ありません


# # 注意　最後は結合するので、列名はpred_rf_fe_imで統一する 
# 

# In[38]:


#予測する
train_data_cluster1['pred_rf_fe_im'] = rf_cluster1_fe_im.predict(X_train_all_cluster1_rf_fe_im)
test_data_cluster1['pred_rf_fe_im'] = rf_cluster1_fe_im.predict(X_test_cluster1_rf_fe_im)


# In[39]:


test_data_cluster1.head()


# In[40]:


#結果表の作成 
train_data_cluster1 = train_data_cluster1.sort_values(['engine_no', 'time_in_cycles'])
test_data_cluster1 = test_data_cluster1.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf'のみ
result_cluster1_rf_fe_im = test_data_cluster1.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_fe_im']]


# In[41]:


result_cluster1_rf_fe_im.head()


# In[42]:


# pred_rf<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_cluster1_rf_fe_im['result'] = result_cluster1_rf_fe_im['pred_rf_fe_im'].map(lambda x: 0 if x > 100 else 1)


# In[43]:


result_cluster1_rf_fe_im.head()
# result_cluster1_rf_fe_im = result_cluster1_rf_fe_im.drop("resul_cluster1t", axis=1)


# In[44]:


#csvで結果の保存
result_cluster1[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/cluster1_submission2.csv', index=False)


# # 上記で読み込んだtest_data_RUL_cluster1を使っていく

# In[45]:


test_data_RUL_cluster1.head()


# # train_allで学習しtestデータで出した予測値と、実際の答えで混合行列を作成

# In[46]:


print((result_cluster1_rf_fe_im['result'] == 1).sum())
print((result_cluster1_rf_fe_im['result'] == 0).sum())


# In[47]:


print((test_data_RUL_cluster1['result'] == 1).sum())
print((test_data_RUL_cluster1['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[48]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL_cluster1['result'],result_cluster1_rf_fe_im['result'],labels=[1, 0] )


# In[49]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)233、真陰性: TN (True-Negative)177、偽陽性: FP (False-Positive)15(故障していないのにしていると予測)、偽陰性: FN (False-Negative)82(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)82を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[50]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL_cluster1['result'],result_cluster1_rf_fe_im['result'],labels=[1, 0] )
print(cr)


# # labelsを付けた時の計算が上記のまとめて表示の1の方　labelsをつけなかったときの計算が下の方

# # #再現率を上げたい　実際の故障のうち故障と予測できたもの

# In[51]:


#特徴量の重要度を算出
feature_importances_cluster1_rf_fe_im = rf_cluster1_fe_im.feature_importances_


feature_importance_cluster1_rf_fe_im =     pd.DataFrame(rf_cluster1_fe_im.feature_importances_, columns=["importance"], index=selected_features)
feature_importance_cluster1_rf_fe_im.sort_values("importance", ascending=False)

#全部足すと100% 縦軸で
#これでは重要度はわかるが、回帰のように上がる下がるなど関係性はわからない


# In[52]:


feature_importance_cluster1_rf_fe_im.sort_values("importance", ascending=False).plot(kind="bar")


# In[53]:


#メモリを軽くする
import gc
 
del result_cluster1
del rf_cluster1
del rf_cluster1_fe_im
gc.collect()


# # クロスバリデーションによるハイパーパラメータのチューニング 上記のfeature importanceで行う

# In[54]:


#クロスバリデーションによるハイパーパラメータのチューニング
rf_cluster1_gscv_fe_im = RandomForestRegressor(random_state=1234)


# In[55]:


params = {'n_estimators': [100, 150, 200], 'max_depth': [50, 100, 150]}


# In[56]:


## 注意: scikit-learnのバージョンが、0.18以降の場合は、scoring='neg_mean_squared_error'とします
rf_cluster1_gscv_fe_im = GridSearchCV(rf_cluster1_gscv_fe_im, param_grid=params, verbose=1,
                    cv=3, scoring='neg_mean_squared_error')

# #注意: scikit-learnのバージョンが、0.17の場合は、scoring='mean_squared_error'とします
# import sklearn
# sklearn.__version__
# gscv = GridSearchCV(rf, param_grid=params, verbose=1,
#                     cv=3, scoring='mean_squared_error')


# In[ ]:


rf_cluster1_gscv_fe_im.fit(X_train_all_cluster1_rf_fe_im, y_train_all_cluster1_rf_fe_im)


# In[ ]:


rf_cluster1_gscv_fe_im.best_params_


# In[ ]:


#ハイパーパラメータチューニング後学習を見てみましょう
rf_cluster1_gscv_fe_im_tu = RandomForestRegressor(n_estimators=200, max_depth=50, random_state=1234)
rf_cluster1_gscv_fe_im_tu.fit(X_train_all_cluster1_rf_fe_im, y_train_all_cluster1_rf_fe_im)


# In[ ]:


#上記のrf_gscv_fe_im_tu_allで学習したモデルでtest_fe_imデータを予測する
train_data_cluster1['pred_rf_gscv_fe_im_tu_all'] = rf_cluster1_gscv_fe_im_tu.predict(X_train_all_cluster1_rf_fe_im)
test_data_cluster1['pred_rf_gscv_fe_im_tu_all'] = rf_cluster1_gscv_fe_im_tu.predict(X_test_cluster1_rf_fe_im)


# In[ ]:


train_data_cluster1.head()


# In[ ]:


test_data_cluster1.head()


# In[ ]:


#結果表の作成 
train_data_cluster1 = train_data_cluster1.sort_values(['engine_no', 'time_in_cycles'])
test_data_cluster1 = test_data_cluster1.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf_gscv_all'のみ
result_cluster1_rf_gscv_fe_im_tu = test_data_cluster1.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_gscv_fe_im_tu_all']]


# In[ ]:


result_cluster1_rf_gscv_fe_im_tu.head()


# In[ ]:


# pred_rf_gscv_all<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_cluster1_rf_gscv_fe_im_tu['result'] = result_cluster1_rf_gscv_fe_im_tu['pred_rf_gscv_fe_im_tu_all'].map(lambda x: 0 if x > 100 else 1)


# In[ ]:


result_cluster1_rf_gscv_fe_im_tu.head()


# In[ ]:


#csvで結果の保存
result_cluster1_rf_gscv_fe_im_tu[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/cluster1_submission3.csv', index=False)


# # 上記で読み込んだtest_data_RUL_cluster1を使っていく

# In[ ]:


test_data_RUL_cluster1.head()


# # train_allで学習しtestデータで出した予測値と、実際の答えで混合行列を作成

# In[ ]:


print((result_cluster1_rf_gscv_fe_im_tu['result'] == 1).sum())
print((result_cluster1_rf_gscv_fe_im_tu['result'] == 0).sum())


# In[ ]:


print((test_data_RUL_cluster1['result'] == 1).sum())
print((test_data_RUL_cluster1['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[ ]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL_cluster1['result'],result_cluster1_rf_gscv_fe_im_tu['result'],labels=[1, 0] )


# In[ ]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)233、真陰性: TN (True-Negative)177、偽陽性: FP (False-Positive)15(故障していないのにしていると予測)、偽陰性: FN (False-Negative)82(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)82を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[ ]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL_cluster1['result'],result_cluster1_rf_gscv_fe_im_tu['result'],labels=[1, 0] )
print(cr)


# # labelsを付けた時の計算が上記のまとめて表示の1の方　labelsをつけなかったときの計算が下の方

# # #再現率を上げたい　実際の故障のうち故障と予測できたもの

# In[ ]:


#特徴量の重要度を算出
feature_importances_cluster1_rf_gscv_fe_im_tu = rf_cluster1_gscv_fe_im_tu_all.feature_importances_


feature_importances_cluster1_rf_gscv_fe_im_tu =     pd.DataFrame(rf_cluster1_gscv_fe_im_tu_all.feature_importances_, columns=["importance"], index=selected_features)
feature_importances_cluster1_rf_gscv_fe_im_tu.sort_values("importance", ascending=False)

#全部足すと100% 縦軸で
#これでは重要度はわかるが、回帰のように上がる下がるなど関係性はわからない


# In[ ]:


feature_importances_cluster1_rf_gscv_fe_im_tu.sort_values("importance", ascending=False).plot(kind="bar")


# # 結果をtrain_data_cluster1_pred、test_data_cluster1_predとして出力する

# In[196]:


#clusterごとに分けたデータの予測結果追加を出力
train_data_cluster1.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster1_pred.csv', index=False)
test_data_cluster1.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster1_pred.csv', index=False)


# In[197]:


#clusterごとに分けたデータの予測結果追加を読み込み
train_data_cluster1_pred = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster1_pred.csv')
test_data_cluster1_pred = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster1_pred.csv')


# # cluster0と1を結合して、test_dataの結果と比較する

# In[11]:


train_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/train_data/train_data.csv')
test_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/test_data/test_data.csv')


# In[2]:


#clusterごとに分けたデータの予測結果追加を読み込み
train_data_cluster0_pred = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster0_pred.csv')
test_data_cluster0_pred = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster0_pred.csv')


# In[3]:


#clusterごとに分けたデータの予測結果追加を読み込み
train_data_cluster1_pred = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster1_pred.csv')
test_data_cluster1_pred = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster1_pred.csv')


# In[4]:


#2つのdataを結合する
train_data_all_pred = train_data_cluster0_pred.append(train_data_cluster1_pred)
test_data_all_pred = test_data_cluster0_pred.append(test_data_cluster1_pred)


# In[5]:


# 2つのataを最初の形に並べ替える
train_data_all_pred = train_data_all_pred.sort_values(['engine_no', 'time_in_cycles'])
test_data_all_pred = test_data_all_pred.sort_values(['engine_no', 'time_in_cycles'])


# In[7]:


#2つのtdataを結合したデータの予測結果追加をを出力
train_data_all_pred.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_all_pred.csv', index=False)
test_data_all_pred.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_all_pred.csv', index=False)


# In[8]:


train_data_all_pred.head()


# # 結合したdataのレコード数が元のdataのレコード数と同じか確認する

# In[12]:


print(train_data.shape)
print(train_data_all_pred.shape)


# In[15]:


print(test_data.shape)
print(test_data_all_pred.shape)


# In[16]:


#結果表の作成 
# train_data_all_pred = train_data_all_pred.sort_values(['engine_no', 'time_in_cycles'])
# test_data_all_pred = test_data_all_pred.sort_values(['engine_no', 'time_in_cycles'])

#test_dataの各engineの最後のサイクルの行のみを表示　列は'engine_no', 'pred_rf_gscv_all'のみ
result_test_data_all_pred = test_data_all_pred.groupby('engine_no').last().reset_index()[['engine_no', 'pred_rf_gscv_fe_im_tu_all']]


# In[17]:


result_test_data_all_pred.head()


# In[18]:


# pred_rf_gscv_fe_im_tu_all<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
result_test_data_all_pred['result'] = result_test_data_all_pred['pred_rf_gscv_fe_im_tu_all'].map(lambda x: 0 if x > 100 else 1)


# In[19]:


result_test_data_all_pred.head()


# In[20]:


#csvで結果の保存
result_test_data_all_pred[['engine_no', 'result']].to_csv('Desktop/インテグ課題/NASA/test_data/result_test_data_all_pred.csv', index=False)


# # test_dataのRULを読み込んで処理

# In[21]:


#testデータのRULを読み込む
test_data_RUL = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/RUL_test.csv')


# In[22]:


test_data_RUL.head()


# In[23]:


#列の順番を入れ替えと削除
test_data_RUL.drop('Unnamed: 0', axis=1)
test_data_RUL = test_data_RUL.loc[:,['engine_no','RUL']]
test_data_RUL.head()


# In[24]:


# RUL<100の時、つまり予測した残りの寿命サイクルが100より小さければ1 それ以外で0とする
test_data_RUL['result'] = test_data_RUL['RUL'].map(lambda x: 0 if x > 100 else 1)


# In[25]:


test_data_RUL.head()


# In[26]:


print((test_data_RUL['result'] == 1).sum())
print((test_data_RUL['result'] == 0).sum())


# # result_test_data_all_predの特徴量選択とパラメータ調整後のデータを使う

# In[27]:


print((result_test_data_all_pred['result'] == 1).sum())
print((result_test_data_all_pred['result'] == 0).sum())


# # 100以内に故障するを1,しないを0としている 故障することを当てたいから、故障する1をpositiveと定義する。

# In[28]:


#混合行列
#1 = Positive,0 = Negativeとする　なぜならば、再現率などがpositiveで計算されるので、故障するをpositiveにする
#labels指定をしないと、左から0,1表記になるので、TPが右下になってしまう。
cm = confusion_matrix(test_data_RUL['result'],result_test_data_all_pred['result'],labels=[1, 0] )


# In[29]:


cm


# # 列が予測の1と0行が実際の1と0つまり、真陽性 : TP (True-Positive)332、真陰性: TN (True-Negative)239、偽陽性: FP (False-Positive)18(故障していないのにしていると予測)、偽陰性: FN (False-Negative)118(実際は故障しているのに故障していないと予測)
# #https://pythondatascience.plavox.info/scikit-learn/%E5%88%86%E9%A1%9E%E7%B5%90%E6%9E%9C%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E8%A9%95%E4%BE%A1
# #FN (False-Negative)118を少なくしたい　つまり実際は故障だが、故障しないと判断してしまっているものを

# In[30]:


#混合行列まとめて表示
cr = classification_report(test_data_RUL['result'],result_test_data_all_pred['result'],labels=[1, 0] )
print(cr)

