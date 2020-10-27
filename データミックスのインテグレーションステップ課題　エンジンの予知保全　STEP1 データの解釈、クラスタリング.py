#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score, f1_score,classification_report
from random import randint, shuffle
import os
import seaborn as sns


# In[2]:


train_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/train_data/train_data.csv')


# In[3]:


test_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/test_data/test_data.csv')


# # データ詳細

# In[4]:


train_data.head()


# In[5]:


test_data.head()


# In[6]:


train_data.describe()


# In[7]:


test_data.describe()


# In[11]:


train_data.shape


# In[12]:


test_data.shape


# # 欠損値の確認

# In[4]:


# train_data
col_names = train_data.columns
for col_name in col_names:
    missing_num = sum(pd.isnull(train_data[col_name]))
    print(col_name,";# missing record:",missing_num)


# In[13]:


# test_data
col_names = test_data.columns
for col_name in col_names:
    missing_num = sum(pd.isnull(test_data[col_name]))
    print(col_name,";# missing record:",missing_num)


# In[5]:


# train_data
# 欠損値のある列を見つける
nan_column = train_data.columns[train_data.isnull().any()].tolist()
print('Columns with all nan: \n' + str(nan_column) + '\n')


# In[5]:


# test_data
# 欠損値のある列を見つける
nan_column = test_data.columns[test_data.isna().any()].tolist()
print('Columns with all nan: \n' + str(nan_column) + '\n')


# # trainデータとtestデータの欠損値の列は同じ

# In[6]:


# 欠損値のある列以外で特徴量選択
selected_features = [x for x in train_data.columns if x not in nan_column]


# # 以下trainデータで作業

# In[7]:


# train_dataの欠損値のある列削除
train_data = train_data[selected_features]


# In[11]:


train_data.head()


# In[9]:


# engine_noの数を確認
train_engine_no = list(train_data.engine_no.drop_duplicates())
train_engine_no


# # train_dataのengine_noは0～708

# # 目的変数RULと時間列time_in_cyclesと各説明変数の関係を可視化する

# In[81]:


# RULとTime_in_cycles折れ線グラフ
fig = plt.figure(figsize=(15, 10))  
for i in range(709):
    x = train_data[train_data['engine_no'] == i].time_in_cycles
    y = train_data[train_data['engine_no'] == i].RUL
    plt.plot(x,y,alpha=0.5,antialiased=True)
    plt.xlabel('time_in_cycles')
    plt.ylabel('RUL')
    plt.title('RUL-time_in_cycles Line graph')


# In[9]:


# RULとtime_in_cycles
train_data.loc[:,['engine_no','RUL','time_in_cycles']].head(10)


# In[10]:


#engine_noの0と1の境目
train_data.loc[334:344,['engine_no','RUL','time_in_cycles']]


# # RULとtime_in_cycles線形の関係になっている。ただし完全な相関関係ではない。RULの最後の値は0で、Time_in_cyclesの最初の値は1になっている。time_in_cyclesが1から進み、engineが故障した時点のRULを0と定義して、そこからRULが1ずつさかのぼっている。

# In[12]:


# RULとop_setting折れ線グラフ
def op_setting_Graph (i):
    y0 = 'op_setting_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(709):
        a = train_data[train_data['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('RUL')
        plt.ylabel('op_setting_'+ str(i))
        plt.title('RUL-op_setting_'+ str(i) + ' Line graph')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data['RUL'].max(),train_data['RUL'].min())


# In[13]:


for i in range(1,4):
    op_setting_Graph(i = i)    


# In[15]:


# RULとsensor折れ線グラフ
def sensor_Graph (i):
    y0 = 'sensor_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(709):
        a = train_data[train_data['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('RUL')
        plt.ylabel('sensor_'+ str(i))
        plt.title('RUL-sensor_'+ str(i) + ' Line graph')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data['RUL'].max(),train_data['RUL'].min())


# In[16]:


for i in range(1,22):
    sensor_Graph(i = i)    


# # センサーのノイズが大きく傾向がつかめない。移動平均で平滑化して傾向を分かりやすくする。

# # 各engineの行数が移動平均数よりも大きくないといけないので、各engineの行数を確認する。

# In[20]:


# engine_noごとのデータを抽出
k = 0
train_data_engine = []
for i in range(709): 
    train_engine = train_data[train_data['engine_no'] == k]
    train_data_engine.append(train_engine)
    k = k + 1


# In[7]:


# 行数が20より小さいengineを見つける
for i in range(709):
    if train_data_engine[i].shape[0] <= 19:
        print('engine_no_' + str(i)  + ': ' +  str(train_data_engine[i].shape[0]))
else:
    print('engine_no_record<=20 is not')        


# # 行数が20より小さいengineはないので、20で移動平均をかける

# In[21]:


train_data_roll_mean = []
# 関数化 engineごとに移動平均をかける。その後、Nanが発生するので、engine_noのNanに元々のengine_noを代入する。
def roll_mean(rollnum):
    for i in range(709):
        train_data_roll_mean_engine = train_data_engine[i].rolling(rollnum).mean()
        train_data_roll_mean.append(train_data_roll_mean_engine)
    for j in range(709):
            train_data_roll_mean[j].loc[:,'engine_no'] = j
    return train_data_roll_mean


# In[22]:


# 後にfor文でグラフを作成するためにリストをDataFrameにする。
def roll_mean_df(i):
    pd.DataFrame(roll_mean(rollnum = 20))
    return pd.DataFrame(train_data_roll_mean[i])
    


# In[ ]:


# DataFrame化したデータを結合する 708個のエンジンすべて
train_data_roll_mean20_df = roll_mean_df(i = 0)
for i in range(1,709):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータ100をcsvで出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df.csv', index=False)


# # 一度にデータを結合するとメモリが足りなくなるので、以下では100ごとにengine_noを区切って、都度csvに出力する。

# In[17]:


# DataFrame化したデータengine_no99までを結合する
train_data_roll_mean20_df = roll_mean_df(i = 0)
for i in range(1,100):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータengine_no99までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df99.csv', index=False)


# In[10]:


# 20で移動平均したデータengine_no99までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df99.csv')


# In[11]:


# DataFrame化したデータengine_no199までを結合する
for i in range(100,200):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))
# 20で移動平均したデータengine_no199までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df199.csv', index=False)


# In[10]:


# 20で移動平均したデータengine_no199までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df199.csv')


# In[11]:


# DataFrame化したデータengine_no299までを結合する
for i in range(200,300):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータengine_no299までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df299.csv', index=False)


# In[10]:


# 20で移動平均したデータengine_no299までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df299.csv')


# In[12]:


# DataFrame化したデータengine_no399までを結合する
for i in range(300,400):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータengine_no399までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df399.csv', index=False)


# In[ ]:


# 20で移動平均したデータengine_no399までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df399.csv')


# In[13]:


# DataFrame化したデータengine_no499までを結合する
for i in range(400,500):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータengine_no499までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df499.csv', index=False)


# In[ ]:


# 20で移動平均したデータengine_no499までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_d499.csv')


# In[14]:


# DataFrame化したデータengine_no599までを結合する
for i in range(500,600):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータengine_no599までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df599.csv', index=False)


# In[23]:


# 20で移動平均したデータengine_no599までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df599.csv')


# In[24]:


# DataFrame化したデータengine_no708(最後)までを結合する
for i in range(600,709):    
    train_data_roll_mean20_df = train_data_roll_mean20_df.append(roll_mean_df(i = i))

# 20で移動平均したデータengine_no708(最後)までを出力
train_data_roll_mean20_df.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df708.csv', index=False)


# In[7]:


# 20で移動平均したデータengine_no708(最後)までを読み込む
train_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df708.csv')


# # 移動平均後のデータのレコード数、カラム数を移動平均前のtrain_dataと等しいかどうか確認する。

# In[8]:


print(train_data.shape)
print(train_data_roll_mean20_df.shape)


# # レコード数、カラム数ともに等しいので問題ない

# # 移動平均後のデータでグラフを作成して傾向を見る

# In[12]:


# RULとop_setting折れ線グラフ
def roll_mean20_op_setting_Graph (i):
    y0 = 'op_setting_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(709):
        a = train_data_roll_mean20_df[train_data_roll_mean20_df['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('RUL')
        plt.ylabel('op_setting_'+ str(i))
        plt.title('Roll_mean RUL-op_setting_'+ str(i) + ' Line graph')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data_roll_mean20_df['RUL'].max(),train_data_roll_mean20_df['RUL'].min())


# In[13]:


for i in range(1,4):
    roll_mean20_op_setting_Graph(i = i)    


# In[14]:


# RULとsensor折れ線グラフ
def roll_mean20_sensor_Graph (i):
    y0 = 'sensor_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(709):
        a = train_data_roll_mean20_df[train_data_roll_mean20_df['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('RUL')
        plt.ylabel('sensor_'+ str(i))
        plt.title('Roll_mean RUL-sensor_'+ str(i) + ' Line graph')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data_roll_mean20_df['RUL'].max(),train_data_roll_mean20_df['RUL'].min())


# In[15]:


for i in range(1,22):
    roll_mean20_sensor_Graph(i = i)    


# # 2つのグループに分かれる傾向があることが確認できる。→クラスタリングを行う

# # クラスタリング

# In[4]:


# StandardScalerで平均0、標準偏差1で標準化する
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[5]:


import sklearn.preprocessing as sp


# In[6]:


# DataFrameをnpに変換する。
train_data_roll_mean20_array = np.array(train_data_roll_mean20_df)


# In[7]:


train_data_roll_mean20_array.shape[0]


# In[8]:


train_data_roll_mean20_array


# In[9]:


# train_data_roll_mean20_arrayをコピーして、標準化に使う
train_data_roll_mean20_array_std = train_data_roll_mean20_array.copy()


# In[10]:


# 列だけ標準化する。
for i in range(1,train_data_roll_mean20_array.shape[1]):
    train_data_roll_mean20_array_std[:,i] = sp.scale(train_data_roll_mean20_array[:,i])


# In[11]:


train_data_roll_mean20_array_std


# In[12]:


# npをデータフレームに変換し直す
train_data_roll_mean20_df_std = pd.DataFrame(train_data_roll_mean20_array_std)


# In[13]:


# 1列目を除くtrain_data_roll_mean20_df_std
target_cols = list(range(1,train_data_roll_mean20_df_std.shape[1]))
train_data_roll_mean20_df_std = train_data_roll_mean20_df_std[target_cols]


# In[14]:


train_data_roll_mean20_df_std


# In[15]:


# 欠損値の置き換え→0にすると後々厄介なので欠損値のある行を削除
train_data_roll_mean20_df_std = train_data_roll_mean20_df_std.dropna(how='any')


# In[16]:


train_data_roll_mean20_df_std


# In[17]:


# クラスタリング
from sklearn.cluster import KMeans


# In[35]:


distortions = []

for i in range(1,15):
    km = KMeans(n_clusters = i,random_state=1234)
    km.fit(train_data_roll_mean20_df_std)
    distortions.append(km.inertia_)
    
plt.plot(range(1,15), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[19]:


# cluster2で再度学習
km = KMeans(n_clusters=2, random_state=1234)


# In[20]:


km.fit(train_data_roll_mean20_df_std)


# In[21]:


#クラスターごとの数を求める
cluster_labels = km.predict(train_data_roll_mean20_df_std)
np.bincount(cluster_labels)


# # Nan行を削除したので、レコード数が元データと会わずに、元データにクラスター番号は付与できない。よって、元データのNanを削除して付与する

# In[22]:


#元データをコピー
train_data_roll_mean20_df_cluster = train_data_roll_mean20_df.copy()


# In[23]:


#Nan行を削除
train_data_roll_mean20_df_cluster = train_data_roll_mean20_df_cluster.dropna(how='any')


# In[24]:


#クラスター番号の列を追加する
train_data_roll_mean20_df_cluster["cluster_labels"] = cluster_labels


# In[25]:


train_data_roll_mean20_df_cluster


# In[28]:


# クラスターを付与したデータを一度出力しておく
train_data_roll_mean20_df_cluster.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df_cluster.csv',index=False)


# In[8]:


#クラスターを付与したデータを読み込む
train_data_roll_mean20_df_cluster = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_roll_mean20_df_cluster.csv')


# # 可視化する

# クラスターごとに変数に格納する

# In[24]:


train_data_roll_mean20_df_cluster0 = train_data_roll_mean20_df_cluster[train_data_roll_mean20_df_cluster['cluster_labels'] == 0]


# In[25]:


train_data_roll_mean20_df_cluster1 = train_data_roll_mean20_df_cluster[train_data_roll_mean20_df_cluster['cluster_labels'] == 1]


# In[26]:


train_data_roll_mean20_df_cluster0


# In[27]:


train_data_roll_mean20_df_cluster1


# In[28]:


# クラスターごとのengine_noをリストに格納
cluster_list0 =  list(train_data_roll_mean20_df_cluster0['engine_no'].drop_duplicates())
cluster_list1 =  list(train_data_roll_mean20_df_cluster1['engine_no'].drop_duplicates())


# In[29]:


print(cluster_list0)
print(cluster_list1)


# In[30]:


print(len(cluster_list0))
print(len(cluster_list1))


# In[31]:


print(cluster_list0)
len(cluster_list0)


# In[32]:


print(cluster_list1)
len(cluster_list1)


# In[33]:


# 2つのクラスターに共通のengine_noを見つける
cluster0_and_1 = set(cluster_list0) & set(cluster_list1)
cluster0_and_1_list = list(cluster0_and_1)
cluster0_and_1_list.sort()
print((cluster0_and_1_list))
len(cluster0_and_1_list)


# In[34]:


type(cluster0_and_1_list)


# # 2つのクラスターに共通のengine_noは[4, 204, 213, 222, 250, 315, 431]の7つ。このengine_noにおいて、それぞれのclusterにおけるレコード数を比較してレコード数の多い方のclusterに統一する。

# In[11]:


# [4, 204, 213, 222, 250, 315, 431]のclusterごとのレコード数を算出
for i in (cluster0_and_1_list):
    print('cluster0のengine_' + str(i) + 'の行数:' + str(train_data_roll_mean20_df_cluster0[train_data_roll_mean20_df_cluster0["engine_no"] == i].shape[0]))
    print('cluster1のengine_' + str(i) + 'の行数:' + str(train_data_roll_mean20_df_cluster1[train_data_roll_mean20_df_cluster1["engine_no"] == i].shape[0]))


# # 共通のengine_noはすべてcluster0が多いのでcluster0に統一する。そのためにcluster1から該当するengine_noを削除する。

# In[35]:


for i in cluster0_and_1_list:
      train_data_roll_mean20_df_cluster1 = train_data_roll_mean20_df_cluster1[train_data_roll_mean20_df_cluster1['engine_no'] != i]
    
    


# In[36]:


#リストも同様に削る　4, 204, 213, 222, 250, 315, 431
cluster_list1.remove(4)
cluster_list1.remove(204)
cluster_list1.remove(213)
cluster_list1.remove(222)
cluster_list1.remove(250)
cluster_list1.remove(315)
cluster_list1.remove(431)


# In[37]:


print(len(cluster_list1))


# # クラスターごと,特徴量ごとに可視化

# In[14]:


# RULとsensor折れ線グラフ
def cluster_sensor_Graph (i):
    y0 = "sensor_"+ str(i)
    fig = plt.figure(figsize=(20, 6)) 
    
    
    for j in cluster_list0:
        a = train_data_roll_mean20_df_cluster0[train_data_roll_mean20_df_cluster0['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.subplot(1, 2, 1)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('RUL')
        plt.ylabel(y0)
        plt.title('RUL-sensor_'+ str(i) + ' Line graph(cluster0)')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data_roll_mean20_df_cluster['RUL'].max(),train_data_roll_mean20_df_cluster['RUL'].min())

        
        
    for j in cluster_list1:
        a = train_data_roll_mean20_df_cluster1[train_data_roll_mean20_df_cluster1['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.subplot(1, 2, 2)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('RUL')
        plt.ylabel(y0)
        plt.title('RUL-sensor_'+ str(i) + ' Line graph(cluster1)')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data_roll_mean20_df_cluster['RUL'].max(),train_data_roll_mean20_df_cluster['RUL'].min())
    


# In[15]:


for i in range(1,22):
    cluster_sensor_Graph(i = i)    


# In[16]:


# RULとpo_setting折れ線グラフ
def cluster_op_setting_Graph (i):
    y0 = "op_setting_"+ str(i)
    fig = plt.figure(figsize=(20, 6)) 
    
    
    for j in cluster_list0:
        a = train_data_roll_mean20_df_cluster0[train_data_roll_mean20_df_cluster0['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.subplot(1, 2, 1)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('RUL')
        plt.ylabel(y0)
        plt.title('RUL-op_setting_'+ str(i) + ' Line graph(cluster0)')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data_roll_mean20_df_cluster['RUL'].max(),train_data_roll_mean20_df_cluster['RUL'].min())

        
        
    for j in cluster_list1:
        a = train_data_roll_mean20_df_cluster1[train_data_roll_mean20_df_cluster1['engine_no'] == j]
        x = a['RUL']
        y = a[y0]
        plt.subplot(1, 2, 2)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('RUL')
        plt.ylabel(y0)
        plt.title('RUL-op_setting_'+ str(i) + ' Line graph(cluster1)')
        # RULを昇順　全体のデータの最大、最小にしないとグラフが見ずらい
        plt.xlim(train_data_roll_mean20_df_cluster['RUL'].max(),train_data_roll_mean20_df_cluster['RUL'].min())


# In[17]:


for i in range(1,4):
    cluster_op_setting_Graph(i = i)    


# # clusterごとのデータフレームをcsvで出力する

# In[38]:


print(cluster_list0)
print(cluster_list1)


# In[39]:


train_data_cluster0 = train_data.query('engine_no == [0, 1, 2, 3, 4, 6, 8, 10, 11, 12, 14, 15, 17, 18, 20, 23, 24, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 73, 75, 76, 77, 79, 80, 81, 83, 85, 88, 89, 92, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 121, 123, 125, 126, 127, 128, 129, 131, 132, 133, 134, 136, 137, 140, 142, 143, 144, 145, 147, 148, 151, 152, 153, 155, 157, 159, 161, 162, 163, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 194, 195, 197, 199, 200, 202, 203, 204, 206, 208, 209, 210, 211, 212, 213, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 232, 235, 237, 239, 240, 241, 242, 243, 245, 246, 249, 250, 251, 253, 255, 256, 257, 258, 260, 261, 265, 266, 267, 268, 269, 270, 271, 272, 273, 275, 276, 277, 278, 279, 280, 281, 283, 284, 285, 286, 287, 289, 290, 291, 292, 294, 296, 297, 298, 299, 300, 301, 305, 306, 307, 308, 309, 312, 313, 315, 316, 317, 318, 319, 321, 322, 324, 325, 328, 329, 330, 331, 333, 335, 337, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 354, 355, 356, 359, 361, 362, 364, 366, 367, 368, 369, 370, 372, 374, 377, 379, 380, 381, 383, 384, 386, 387, 388, 391, 392, 396, 399, 400, 401, 402, 403, 406, 407, 408, 409, 411, 412, 413, 414, 417, 419, 420, 422, 424, 427, 428, 430, 431, 432, 433, 434, 435, 436, 437, 439, 440, 441, 442, 443, 446, 447, 448, 449, 450, 451, 452, 454, 455, 456, 458, 461, 462, 464, 465, 466, 467, 469, 470, 471, 472, 474, 476, 479, 480, 483, 485, 486, 487, 488, 490, 491, 492, 493, 494, 495, 497, 498, 499, 500, 501, 503, 504, 505, 506, 508, 509, 510, 511, 512, 514, 515, 516, 517, 518, 520, 522, 523, 524, 525, 526, 527, 530, 531, 532, 534, 535, 537, 538, 540, 542, 543, 544, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 574, 576, 578, 579, 580, 582, 583, 584, 585, 587, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599, 602, 603, 604, 605, 607, 608, 609, 610, 612, 614, 615, 616, 618, 619, 620, 622, 627, 628, 629, 630, 631, 632, 636, 637, 638, 640, 641, 642, 646, 648, 649, 650, 651, 652, 653, 655, 657, 658, 659, 660, 662, 663, 666, 667, 668, 669, 670, 675, 676, 677, 678, 681, 682, 684, 685, 686, 687, 689, 690, 691, 692, 693, 694, 695, 696, 698, 699, 700, 701, 702, 704, 707, 708]')


# In[40]:


train_data_cluster1 = train_data.query('engine_no == [5, 7, 9, 13, 16, 19, 21, 22, 25, 30, 34, 53, 54, 56, 57, 59, 69, 70, 71, 72, 74, 78, 82, 84, 86, 87, 90, 91, 93, 101, 106, 118, 122, 124, 130, 135, 138, 139, 141, 146, 149, 150, 154, 156, 158, 160, 167, 178, 183, 185, 193, 196, 198, 201, 205, 207, 214, 215, 216, 230, 231, 233, 234, 236, 238, 244, 247, 248, 252, 254, 259, 262, 263, 264, 274, 282, 288, 293, 295, 302, 303, 304, 310, 311, 314, 320, 323, 326, 327, 332, 334, 336, 338, 347, 352, 353, 357, 358, 360, 363, 365, 371, 373, 375, 376, 378, 382, 385, 389, 390, 393, 394, 395, 397, 398, 404, 405, 410, 415, 416, 418, 421, 423, 425, 426, 429, 438, 444, 445, 453, 457, 459, 460, 463, 468, 473, 475, 477, 478, 481, 482, 484, 489, 496, 502, 507, 513, 519, 521, 528, 529, 533, 536, 539, 541, 545, 557, 573, 575, 577, 581, 586, 588, 598, 600, 601, 606, 611, 613, 617, 621, 623, 624, 625, 626, 633, 634, 635, 639, 643, 644, 645, 647, 654, 656, 661, 664, 665, 671, 672, 673, 674, 679, 680, 683, 688, 697, 703, 705, 706]')


# In[46]:


# clusterに分けたレコード数を足したものが、分ける前のレコード数と同じか確認する。
print(train_data_cluster0.shape)
print(train_data_cluster1.shape)
print(train_data.shape)


# # cluster0と1に分けたデータをCSVファイルで保存する

# In[47]:


# csvで出力
train_data_cluster0.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster0.csv',index=False)
train_data_cluster1.to_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster1.csv',index=False)


# In[10]:


# csvを読み込み
train_data_cluster0 = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster0.csv')
train_data_cluster1 = pd.read_csv('Desktop/インテグ課題/NASA/train_data/train_data_cluster1.csv',index=False)


# # test_dataでも同様に作業する

# In[2]:


test_data = pd.read_csv('Downloads/インテグ課題/NASA/prehackathonsup/test_data/test_data.csv')


# In[6]:


# test_data
# 欠損値のある列を見つける
nan_column = test_data.columns[test_data.isna().any()].tolist()
print('Columns with all nan: \n' + str(nan_column) + '\n')


# In[7]:


# 欠損値のある列以外で特徴量選択
selected_features = [x for x in test_data.columns if x not in nan_column]


# In[8]:


# test_dataの欠損値のある列削除
test_data = test_data[selected_features]


# In[30]:


test_data.head()


# In[9]:


# engine_noの数を確認
test_engine_no = list(test_data.engine_no.drop_duplicates())
test_engine_no


# # test_dataのengine_noは0～706

# # testdDataにはRULがないので、time_in_cyclesとその他の説明変数の関係を可視化する

# In[22]:


# time_in_cyclesとop_setting折れ線グラフ
def op_setting_Graph (i):
    y0 = 'op_setting_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(707):
        a = test_data[test_data['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('time_in_cycles')
        plt.ylabel('op_setting_'+ str(i))
        plt.title('time_in_cycles-op_setting_'+ str(i) + ' Line graph')
        


# In[23]:


for i in range(1,4):
    op_setting_Graph(i = i)    


# In[24]:


# time_in_cyclesとsensor折れ線グラフ
def sensor_Graph (i):
    y0 = 'sensor_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(707):
        a = test_data[test_data['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('time_in_cycles')
        plt.ylabel('sensor_'+ str(i))
        plt.title('time_in_cycles-sensor_'+ str(i) + ' Line graph')


# In[25]:


for i in range(1,22):
    sensor_Graph(i = i)    


# # センサーのノイズが大きく傾向がつかめない。移動平均で平滑化して傾向を分かりやすくする。

# # 各engineの行数が移動平均数よりも大きくないといけないので、各engineの行数を確認する。

# In[6]:


# engine_noごとのデータを抽出
k = 0
test_data_engine = []
for i in range(707): 
    test_engine = test_data[test_data['engine_no'] == k]
    test_data_engine.append(test_engine)
    k = k + 1


# In[39]:


# 行数が20より小さいengineを見つける
for i in range(707):
    if test_data_engine[i].shape[0] <= 19:
        print('engine_no_' + str(i)  + ': ' +  str(test_data_engine[i].shape[0]))
else:
    print('engine_no_record<=20 is not')        


# # レコード数が20より小さいengineがあり、最小のレコード数が19なので、移動平均15で平滑化する

# In[7]:


test_data_roll_mean = []
# 関数化 engineごとに移動平均をかける。その後、Nanが発生するので、engine_noのNanに元々のengine_noを代入する。
def roll_mean(rollnum):
    for i in range(707):
        test_data_roll_mean_engine = test_data_engine[i].rolling(rollnum).mean()
        test_data_roll_mean.append(test_data_roll_mean_engine)
    for j in range(707):
            test_data_roll_mean[j].loc[:,'engine_no'] = j
    return test_data_roll_mean


# In[8]:


# 後にfor文でグラフを作成するためにリストをDataFrameにする。
def roll_mean_df(i):
    pd.DataFrame(roll_mean(rollnum = 15))
    return pd.DataFrame(test_data_roll_mean[i])
    


# In[ ]:


# DataFrame化したデータを結合する 706個のエンジンすべて
test_data_roll_mean15_df = roll_mean_df(i = 0)
for i in range(1,707):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータをcsvで出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df.csv', index=False)


# # 一度にデータを結合するとメモリが足りなくなるので、以下では100ごとにengine_noを区切って、都度csvに出力している。

# In[ ]:


# DataFrame化したデータengine_no99までを結合する
test_data_roll_mean15_df = roll_mean_df(i = 0)
for i in range(1,100):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータengine_no99までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df99.csv', index=False)


# In[10]:


# 15で移動平均したデータengine_no99までを読み込む
test_data_roll_mean15_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df99.csv')


# In[11]:


# DataFrame化したデータengine_no199までを結合する
for i in range(100,200):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))
# 20で移動平均したデータengine_no199までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df199.csv', index=False)


# In[42]:


# 15で移動平均したデータengine_no199までを読み込む
test_data_roll_mean15_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df199.csv')


# In[12]:


# DataFrame化したデータengine_no299までを結合する
for i in range(200,300):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータengine_no299までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df299.csv', index=False)


# In[ ]:


# 15で移動平均したデータengine_no299までを読み込む
test_data_roll_mean15_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df299.csv')


# In[13]:


# DataFrame化したデータengine_no399までを結合する
for i in range(300,400):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータengine_no399までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df399.csv', index=False)


# In[ ]:


# 15で移動平均したデータengine_no399までを読み込む
test_data_roll_mean15_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df399.csv')


# In[14]:


# DataFrame化したデータengine_no499までを結合する
for i in range(400,500):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータengine_no499までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df499.csv', index=False)


# In[ ]:


# 20で移動平均したデータengine_no499までを読み込む
test_data_roll_mean20_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean20_d499.csv')


# In[15]:


# DataFrame化したデータengine_no599までを結合する
for i in range(500,600):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータengine_no599までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df599.csv', index=False)


# In[ ]:


# 20で移動平均したデータengine_no499までを読み込む
test_data_roll_mean15_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df599.csv')


# In[16]:


# DataFrame化したデータengine_no706(最後)までを結合する
for i in range(600,707):    
    test_data_roll_mean15_df = test_data_roll_mean15_df.append(roll_mean_df(i = i))

# 15で移動平均したデータengine_no706(最後)までを出力
test_data_roll_mean15_df.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df706.csv', index=False)


# In[2]:


# 20で移動平均したデータengine_no706(最後)までを読み込む
test_data_roll_mean15_df = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df706.csv')


# # 移動平均後のデータのレコード数、カラム数を移動平均前のtest_dataと等しいかどうか確認する。

# In[7]:


print(test_data.shape)
print(test_data_roll_mean15_df.shape)


# # レコード数、カラム数ともに等しいので問題ない

# # 移動平均後のデータでグラフを作成して傾向を見る

# In[8]:


# RULとop_setting折れ線グラフ
def roll_mean15_op_setting_Graph (i):
    y0 = 'op_setting_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(709):
        a = test_data_roll_mean15_df[test_data_roll_mean15_df['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('time_in_cycles')
        plt.ylabel('op_setting_'+ str(i))
        plt.title('time_in_cycles-op_setting_'+ str(i) + ' Line graph')
        


# In[9]:


for i in range(1,4):
    roll_mean15_op_setting_Graph(i = i)    


# In[10]:


# RULとsensor折れ線グラフ
def roll_mean15_sensor_Graph (i):
    y0 = 'sensor_'+ str(i)
    fig = plt.figure(figsize=(10, 6)) 
    for j in range(709):
        a = test_data_roll_mean15_df[test_data_roll_mean15_df['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.plot(x,y,alpha=0.5)      
        plt.xlabel('time_in_cycles')
        plt.ylabel('sensor_'+ str(i))
        plt.title('time_in_cycles-sensor_'+ str(i) + ' Line graph')
        


# In[11]:


for i in range(1,22):
    roll_mean15_sensor_Graph(i = i)    


# # 2つのグループに分かれる傾向があることが確認できる。→クラスタリングを行う

# # クラスタリング

# In[12]:


# StandardScalerで平均0、標準偏差1で標準化する
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[13]:


import sklearn.preprocessing as sp


# In[14]:


test_data_roll_mean15_df


# In[15]:


# DataFrameをnpにする。
test_data_roll_mean15_array = np.array(test_data_roll_mean15_df)


# In[16]:


# レコード数確認
test_data_roll_mean15_array.shape[0]


# In[18]:


test_data_roll_mean15_array


# In[19]:


# test_data_roll_mean15_arrayをコピーしてstdをつけて、標準化に使う
test_data_roll_mean15_array_std = test_data_roll_mean15_array.copy()


# In[20]:


# 列だけ標準化する。
for i in range(1,test_data_roll_mean15_array.shape[1]):
    test_data_roll_mean15_array_std[:,i] = sp.scale(test_data_roll_mean15_array[:,i])


# In[21]:


test_data_roll_mean15_array_std


# In[22]:


# データフレームに直す
test_data_roll_mean15_df_std = pd.DataFrame(test_data_roll_mean15_array_std)


# In[23]:


# 1列目を除く
target_cols = list(range(1,test_data_roll_mean15_df_std.shape[1]))
test_data_roll_mean15_df_std = test_data_roll_mean15_df_std[target_cols]


# In[24]:


test_data_roll_mean15_df_std


# In[25]:


# 欠損値の置き換え→0にすると後々厄介なので欠損値のある行を削除
test_data_roll_mean15_df_std = test_data_roll_mean15_df_std.dropna(how='any')


# In[26]:


test_data_roll_mean15_df_std


# In[27]:


#クラスタリング
from sklearn.cluster import KMeans


# In[28]:


distortions = []

for i in range(1,15):
    km = KMeans(n_clusters = i,random_state=1234)
    km.fit(test_data_roll_mean15_df_std)
    distortions.append(km.inertia_)
    
plt.plot(range(1,15), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[29]:


# cluster2で再度学習
km = KMeans(n_clusters=2, random_state=1234)


# In[31]:


km.fit(test_data_roll_mean15_df_std)


# In[32]:


#クラスターごとの数を求める
cluster_labels = km.predict(test_data_roll_mean15_df_std)
np.bincount(cluster_labels)


# # Nan行を削除したので、レコード数が元データと会わずに、元データにクラスター番号は付与できない。よって、元データのNanを削除して付与する

# In[33]:


#元データをコピー
test_data_roll_mean15_df_cluster = test_data_roll_mean15_df.copy()


# In[34]:


#Nan行を削除
test_data_roll_mean15_df_cluster = test_data_roll_mean15_df_cluster.dropna(how='any')


# In[35]:


#クラスター番号の列を追加する
test_data_roll_mean15_df_cluster["cluster_labels"] = cluster_labels


# In[36]:


test_data_roll_mean15_df_cluster


# In[40]:


#クラスターを付与したデータを出力
test_data_roll_mean15_df_cluster.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df_cluster.csv',index=False)


# In[9]:


#クラスターを付与したデータを読み込む
test_data_roll_mean15_df_cluster = pd.read_csv('Desktop/インテグ課題/NASA/test_data/test_data_roll_mean15_df_cluster.csv')


# # 可視化する

# クラスターごとに変数に格納

# In[13]:


test_data_roll_mean15_df_cluster0 = test_data_roll_mean15_df_cluster[test_data_roll_mean15_df_cluster['cluster_labels'] == 0]


# In[14]:


test_data_roll_mean15_df_cluster1 = test_data_roll_mean15_df_cluster[test_data_roll_mean15_df_cluster['cluster_labels'] == 1]


# In[15]:


test_data_roll_mean15_df_cluster0


# In[16]:


test_data_roll_mean15_df_cluster1


# In[17]:


#クラスターごとのengine_noをリストに格納
cluster_list0 =  list(test_data_roll_mean15_df_cluster0['engine_no'].drop_duplicates())
cluster_list1 =  list(test_data_roll_mean15_df_cluster1['engine_no'].drop_duplicates())


# In[18]:


print(cluster_list0)
print(cluster_list1)


# In[19]:


print(len(cluster_list0))
print(len(cluster_list1))


# In[20]:


print(cluster_list0)
len(cluster_list0)


# In[21]:


print(cluster_list1)
len(cluster_list1)


# In[22]:


# 2つのクラスターに共通のengine_noを見つける
cluster0_and_1 = set(cluster_list0) & set(cluster_list1)
cluster0_and_1_list = list(cluster0_and_1)
cluster0_and_1_list.sort()
print((cluster0_and_1_list))
len(cluster0_and_1_list)


# # 2つのクラスターに共通のengine_noは[0, 51, 69, 105, 150, 174, 198, 217, 231, 336, 341, 342, 356, 388, 411, 452, 455, 468, 515, 527, 547, 602, 648, 652, 655, 704]。このengine_noにおいて、それぞれのclusterにおけるレコード数を比較してレコード数の多い方のclusterに統一する。

# In[57]:


# [[0, 51, 69, 105, 150, 174, 198, 217, 231, 336, 341, 342, 356, 388, 411, 452, 455, 468, 515, 527, 547, 602, 648, 652, 655, 704]のclusterごとのレコード数を算出
for i in cluster0_and_1_list:
    if test_data_roll_mean15_df_cluster0[test_data_roll_mean15_df_cluster0["engine_no"] == i].shape[0] > test_data_roll_mean15_df_cluster1[test_data_roll_mean15_df_cluster1["engine_no"] == i].shape[0]:
        print('engine_' + str(i) + ' cluster0が多い:' + 'レコード数:' + str(test_data_roll_mean15_df_cluster0[test_data_roll_mean15_df_cluster0["engine_no"] == i].shape[0]))
    elif test_data_roll_mean15_df_cluster0[test_data_roll_mean15_df_cluster0["engine_no"] == i].shape[0] < test_data_roll_mean15_df_cluster1[test_data_roll_mean15_df_cluster1["engine_no"] == i].shape[0]:
         print('engine_' + str(i) + ' cluster1が多い:' + 'レコード数:' + str(test_data_roll_mean15_df_cluster1[test_data_roll_mean15_df_cluster1["engine_no"] == i].shape[0]))

  


# # 共通のengine_noはすべてcluster0が多いのでcluster0に統一する。そのためにcluster1から該当するengine_noを削除する。

# In[23]:


# cluster1から該当するengine_noを削除する
for i in cluster0_and_1_list:
      test_data_roll_mean15_df_cluster1 = test_data_roll_mean15_df_cluster1[test_data_roll_mean15_df_cluster1['engine_no'] != i]
    
    


# In[24]:


# リストも同様に削る
for i in cluster0_and_1_list:
    cluster_list1.remove(i)


# In[25]:


# cluster0とcluster1のengineを足した数がclusterに分ける前のengineと同じか確認する
print(len(list(test_data_roll_mean15_df_cluster['engine_no'].drop_duplicates())))
print(len(cluster_list0) + len(cluster_list1))


# # engine数が等しいので、過不足なくcluster0とcluster1に分けられている

# # クラスターごと,特徴量ごとに可視化

# In[40]:


# RULとsensor折れ線グラフ
def cluster_sensor_Graph (i):
    y0 = "sensor_"+ str(i)
    fig = plt.figure(figsize=(20, 6)) 
    
    
    for j in cluster_list0:
        a = test_data_roll_mean15_df_cluster0[test_data_roll_mean15_df_cluster0['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.subplot(1, 2, 1)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('time_in_cycles')
        plt.ylabel(y0)
        plt.title('time_in_cycles-sensor_'+ str(i) + ' Line graph(cluster0)')
       
        
        
    for j in cluster_list1:
        a = test_data_roll_mean15_df_cluster1[test_data_roll_mean15_df_cluster1['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.subplot(1, 2, 2)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('time_in_cycles')
        plt.ylabel(y0)
        plt.title('time_in_cycles-sensor_'+ str(i) + ' Line graph(cluster1)')
        


# In[41]:


for i in range(1,22):
    cluster_sensor_Graph(i = i)    


# In[46]:


# RULとop_settingの折れ線グラフ
def cluster_op_setting_Graph (i):
    y0 = "op_setting_"+ str(i)
    fig = plt.figure(figsize=(20, 6)) 
    
    
    for j in cluster_list0:
        a = test_data_roll_mean15_df_cluster0[test_data_roll_mean15_df_cluster0['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.subplot(1, 2, 1)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('time_in_cycles')
        plt.ylabel(y0)
        plt.title('time_in_cycles-op_setting_'+ str(i) + ' Line graph(cluster0)')
      
        
        
    for j in cluster_list1:
        a = test_data_roll_mean15_df_cluster1[test_data_roll_mean15_df_cluster1['engine_no'] == j]
        x = a['time_in_cycles']
        y = a[y0]
        plt.subplot(1, 2, 2)
        plt.plot(x, y,alpha=0.5,label=y0)
        plt.xlabel('time_in_cycles')
        plt.ylabel(y0)
        plt.title('time_in_cycles-op_setting_'+ str(i) + ' Line graph(cluster1)')
        


# In[47]:


for i in range(1,4):
    cluster_op_setting_Graph(i = i)    


# In[61]:


print(cluster_list0)
print(cluster_list1)


# In[26]:


test_data_cluster0 = test_data.query('engine_no == [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 34, 35, 36, 38, 39, 40, 41, 43, 45, 46, 47, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 65, 66, 67, 69, 71, 72, 74, 75, 76, 77, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 99, 100, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 115, 117, 119, 121, 122, 123, 124, 125, 126, 128, 130, 131, 132, 133, 135, 137, 138, 139, 140, 142, 143, 144, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 162, 163, 164, 165, 166, 167, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 192, 193, 194, 197, 198, 203, 205, 206, 209, 210, 211, 212, 213, 214, 217, 218, 219, 222, 223, 224, 225, 226, 228, 229, 231, 232, 234, 235, 237, 239, 240, 241, 242, 243, 244, 246, 247, 249, 250, 252, 253, 255, 256, 257, 261, 262, 263, 264, 265, 267, 268, 269, 272, 273, 275, 276, 279, 280, 281, 283, 284, 285, 286, 287, 288, 289, 293, 294, 295, 296, 297, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 312, 313, 315, 317, 318, 319, 320, 322, 324, 325, 326, 328, 329, 330, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 347, 351, 352, 354, 356, 357, 359, 360, 361, 362, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 382, 385, 386, 387, 388, 391, 392, 394, 395, 396, 397, 398, 399, 401, 402, 403, 404, 405, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 422, 424, 426, 428, 431, 432, 433, 435, 436, 438, 439, 440, 441, 442, 443, 444, 447, 448, 449, 450, 451, 452, 454, 455, 459, 460, 462, 463, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 487, 489, 490, 491, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 507, 508, 510, 511, 512, 513, 514, 515, 522, 523, 524, 526, 527, 530, 531, 532, 533, 535, 536, 537, 539, 540, 541, 543, 544, 545, 547, 548, 550, 551, 553, 554, 555, 557, 559, 560, 562, 563, 564, 565, 566, 567, 568, 569, 571, 572, 574, 575, 576, 577, 579, 580, 581, 582, 583, 585, 586, 587, 589, 591, 594, 595, 596, 597, 598, 599, 601, 602, 603, 604, 606, 607, 608, 609, 610, 611, 614, 616, 617, 619, 621, 622, 623, 625, 626, 628, 629, 630, 631, 632, 634, 635, 636, 637, 639, 641, 642, 643, 645, 646, 648, 649, 650, 651, 652, 653, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668, 669, 671, 672, 673, 674, 676, 678, 679, 681, 682, 687, 688, 689, 691, 693, 698, 699, 700, 701, 703, 704, 705]')


# In[27]:


test_data_cluster1 = test_data.query('engine_no == [9, 10, 15, 18, 26, 31, 32, 37, 42, 44, 48, 52, 53, 61, 62, 63, 64, 68, 70, 73, 78, 79, 80, 82, 95, 98, 101, 108, 109, 114, 116, 118, 120, 127, 129, 134, 136, 141, 145, 148, 159, 160, 161, 168, 169, 170, 172, 184, 191, 195, 196, 199, 200, 201, 202, 204, 207, 208, 215, 216, 220, 221, 227, 230, 233, 236, 238, 245, 248, 251, 254, 258, 259, 260, 266, 270, 271, 274, 277, 278, 282, 290, 291, 292, 298, 299, 310, 311, 314, 316, 321, 323, 327, 331, 343, 344, 345, 346, 348, 349, 350, 353, 355, 358, 363, 367, 373, 381, 383, 384, 389, 390, 393, 400, 406, 407, 410, 420, 423, 425, 427, 429, 430, 434, 437, 445, 446, 453, 456, 457, 458, 461, 464, 465, 480, 486, 488, 492, 493, 494, 506, 509, 516, 517, 518, 519, 520, 521, 525, 528, 529, 534, 538, 542, 546, 549, 552, 556, 558, 561, 570, 573, 578, 584, 588, 590, 592, 593, 600, 605, 612, 613, 615, 618, 620, 624, 627, 633, 638, 640, 644, 647, 654, 665, 670, 675, 677, 680, 683, 684, 685, 686, 690, 692, 694, 695, 696, 697, 702, 706]')


# In[28]:


print(test_data_cluster0.shape)
print(test_data_cluster1.shape)
print(test_data.shape)


# # cluster0と1に分けたデータをCSVファイルで保存する

# In[29]:


# csvで出力
test_data_cluster0.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster0.csv',index=False)
test_data_cluster1.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_cluster1.csv',index=False)


# In[10]:


# csvを読み込み
test_data_cluster0 = pd.read_csv('Desktop/インテグ課題/NASA/test_data/train_data_cluster0.csv',index=False)
test_data_cluster1 = pd.read_csv('Desktop/インテグ課題/NASA/test_data/train_data_cluster1.csv',index=False)


# # testデータのRUL（答え）もclusterごとに分けて、他必要な処理を行う 

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


# In[21]:


test_data_RUL_cluster0 = test_data_RUL.query('engine_no == [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 34, 35, 36, 38, 39, 40, 41, 43, 45, 46, 47, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 65, 66, 67, 69, 71, 72, 74, 75, 76, 77, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 99, 100, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 115, 117, 119, 121, 122, 123, 124, 125, 126, 128, 130, 131, 132, 133, 135, 137, 138, 139, 140, 142, 143, 144, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 162, 163, 164, 165, 166, 167, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 192, 193, 194, 197, 198, 203, 205, 206, 209, 210, 211, 212, 213, 214, 217, 218, 219, 222, 223, 224, 225, 226, 228, 229, 231, 232, 234, 235, 237, 239, 240, 241, 242, 243, 244, 246, 247, 249, 250, 252, 253, 255, 256, 257, 261, 262, 263, 264, 265, 267, 268, 269, 272, 273, 275, 276, 279, 280, 281, 283, 284, 285, 286, 287, 288, 289, 293, 294, 295, 296, 297, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 312, 313, 315, 317, 318, 319, 320, 322, 324, 325, 326, 328, 329, 330, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 347, 351, 352, 354, 356, 357, 359, 360, 361, 362, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 382, 385, 386, 387, 388, 391, 392, 394, 395, 396, 397, 398, 399, 401, 402, 403, 404, 405, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 422, 424, 426, 428, 431, 432, 433, 435, 436, 438, 439, 440, 441, 442, 443, 444, 447, 448, 449, 450, 451, 452, 454, 455, 459, 460, 462, 463, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 487, 489, 490, 491, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 507, 508, 510, 511, 512, 513, 514, 515, 522, 523, 524, 526, 527, 530, 531, 532, 533, 535, 536, 537, 539, 540, 541, 543, 544, 545, 547, 548, 550, 551, 553, 554, 555, 557, 559, 560, 562, 563, 564, 565, 566, 567, 568, 569, 571, 572, 574, 575, 576, 577, 579, 580, 581, 582, 583, 585, 586, 587, 589, 591, 594, 595, 596, 597, 598, 599, 601, 602, 603, 604, 606, 607, 608, 609, 610, 611, 614, 616, 617, 619, 621, 622, 623, 625, 626, 628, 629, 630, 631, 632, 634, 635, 636, 637, 639, 641, 642, 643, 645, 646, 648, 649, 650, 651, 652, 653, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668, 669, 671, 672, 673, 674, 676, 678, 679, 681, 682, 687, 688, 689, 691, 693, 698, 699, 700, 701, 703, 704, 705]')


# In[22]:


test_data_RUL_cluster1 = test_data_RUL.query('engine_no == [9, 10, 15, 18, 26, 31, 32, 37, 42, 44, 48, 52, 53, 61, 62, 63, 64, 68, 70, 73, 78, 79, 80, 82, 95, 98, 101, 108, 109, 114, 116, 118, 120, 127, 129, 134, 136, 141, 145, 148, 159, 160, 161, 168, 169, 170, 172, 184, 191, 195, 196, 199, 200, 201, 202, 204, 207, 208, 215, 216, 220, 221, 227, 230, 233, 236, 238, 245, 248, 251, 254, 258, 259, 260, 266, 270, 271, 274, 277, 278, 282, 290, 291, 292, 298, 299, 310, 311, 314, 316, 321, 323, 327, 331, 343, 344, 345, 346, 348, 349, 350, 353, 355, 358, 363, 367, 373, 381, 383, 384, 389, 390, 393, 400, 406, 407, 410, 420, 423, 425, 427, 429, 430, 434, 437, 445, 446, 453, 456, 457, 458, 461, 464, 465, 480, 486, 488, 492, 493, 494, 506, 509, 516, 517, 518, 519, 520, 521, 525, 528, 529, 534, 538, 542, 546, 549, 552, 556, 558, 561, 570, 573, 578, 584, 588, 590, 592, 593, 600, 605, 612, 613, 615, 618, 620, 624, 627, 633, 638, 640, 644, 647, 654, 665, 670, 675, 677, 680, 683, 684, 685, 686, 690, 692, 694, 695, 696, 697, 702, 706]')


# In[23]:


#clusterごとに分けたデータを出力
test_data_RUL_cluster0.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_RUL_cluster0.csv',index=False)
test_data_RUL_cluster1.to_csv('Desktop/インテグ課題/NASA/test_data/test_data_RUL_cluster1.csv',index=False)

