# coding: utf-8
#####多項ロジットモデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from numpy.random import *
from scipy import optimize
import seaborn as sns


####データの発生####
##データの設定
N = 3000   #サンプル数
s = 5   #選択肢数


##説明変数の発生
#選択肢ごとに異なる変数
PRICE = uniform(0.6, 1.0, (N, s)) 
PRICE = np.zeros((N, s))
DISP = np.zeros((N, s))
CAMP = np.zeros((N, s))

for i in range(0, 5):
    #パラメータの設定
    price_lower = uniform(0.5, 0.8)
    price_upper = uniform(0.9, 1.25)
    disp = uniform(0.25, 0.5)
    camp = uniform(0.2, 0.45)
    
    #マーケティング変数の発生
    PRICE[:, i] = uniform(price_lower, price_upper, N)
    DISP[:, i] = binomial(1, disp, N)
    CAMP[:, i] = binomial(1, camp, N)



#選択肢ごとに共通の説明変数
ROY = randn(N)   #カテゴリーロイヤルティー
FAMILY = binomial(1, 0.4, N)   #家族構成


##説明変数をベクトル形式に変更
#IDの設定
id = np.array([])
select = np.array([])
for i in range(1, N+1):
    id_vec =  np.matlib.repmat(i, s, 1)
    select_vec = np.arange(1, s+1, 1)
    id = np.append(id, np.ravel(id_vec))
    select = np.append(select, select_vec)

#IDを結合
ID = np.concatenate((np.arange(1, s*N+1, 1)[:, np.newaxis], id[:, np.newaxis], select[:, np.newaxis]), axis=1)
ID_df = pd.DataFrame(np.round(ID, 0))
ID_df.columns = ["no", "id", "select"]


#切片の設定
diag_ind = np.ravel(np.diag(np.ravel(np.matlib.repmat(1, s, 1))))
diag_vec = np.ravel(np.matlib.repmat(diag_ind, N, 1))
intercept = np.reshape(diag_vec, (N*s, s))
intercept = intercept[:, range(s-1)]

#マーケティング変数のベクトル化
PRICE_v = np.ravel(PRICE)[:, np.newaxis]
DISP_v = np.ravel(DISP)[:, np.newaxis]
CAMP_v = np.ravel(CAMP)[:, np.newaxis]


#選択肢で説明変数が変わらない変数のベクトル化
ROY_v = np.zeros((N*s, s))
FAMILY_v = np.zeros((N*s, s))

for i in range(ROY.shape[0]):
    ROY_v[ID[:, 1]==i+1, :] = np.diag(np.ravel(np.matlib.repmat(ROY[i], s, 1)))
    FAMILY_v[ID[:, 1]==i+1, :] = np.diag(np.ravel(np.matlib.repmat(FAMILY[i], s, 1)))

ROY_v = ROY_v[:, range(s-1)]
FAMILY_v = FAMILY_v[:, range(s-1)]


#データを結合
Data = np.concatenate((intercept, PRICE_v, DISP_v, CAMP_v, ROY_v, FAMILY_v), axis=1)
X = pd.DataFrame(Data)
np


####応答変数の発生####
##パラメータベクトルの発生
b0 = np.array((2.9, 2.0, 1.4, 0.8))
b1 = np.array(uniform(-4.3, -3.5))[np.newaxis]
b2 = uniform(1.5, 2.1, 2)
b3 = uniform(0, 1.3, s-1)
b4 = uniform(-1.1, 1.8, s-1)
b = np.concatenate((b0, b1, b2, b3, b4), axis=0)


##ロジットと確率の計算
#ロジットの計算
logit = np.dot(Data, b)
logit_m = np.reshape(logit, (N, s))   #行列化

#確率の計算
Pr = np.exp(logit_m)/np.matlib.repmat(np.sum(np.exp(logit_m), axis=1), s, 1).T
Pr_df = pd.DataFrame(Pr)

#応答変数の発生
Y = np.zeros((N, s))
for i in range(Y.shape[0]):
    Y[i, :] = multinomial(1, Pr[i], 1)
    
#集計
print(np.sum(Y, axis=0))
np.round(pd.DataFrame(np.concatenate((Y, Pr), axis=1)), 3)


####最尤法で多項ロジットモデルを推定####
##多項ロジットモデルの対数尤度の関数
def define_likelihood(beta, Y, Data):
    
    #効用関数の計算
    U = np.dot(Data, beta)
    U_m = np.reshape(U, (N, s))   #行列化

    #確率の計算
    Pr = np.exp(U_m) / np.matlib.repmat(np.sum(np.exp(U_m), axis=1), s, 1).T

    #対数尤度の計算
    LF = Y * np.log(Pr)
    LL = -np.sum(LF)
    return(LL)


##対数尤度を最大化する
theta0 = uniform(0.2, 1.3, Data.shape[1])   #初期値の設定
res = optimize.minimize(define_likelihood, theta0, args=(Y, Data), method='Powell')

#推定結果
print(-res.fun)   #最大化された対数尤度
round(pd.DataFrame(np.array((res.x, b))), 3)   #推定されたパラメータと真のパラメータの比較



