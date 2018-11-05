#####打ち切りデータのモデリング#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib
import scipy.linalg
from scipy.special import gammaln
from scipy.misc import factorial
from pandas.tools.plotting import scatter_matrix
from numpy.random import *
from scipy import optimize
from scipy.stats import norm
import seaborn as sns


#####片側打ち切りモデル#####
####データの発生####
n = 10000   #サンプル数
p = 15   #説明変数数
b = uniform(-1.5, 4.5, p)   #回帰係数のパラメータ
b0 = 8.4   #切片
sigma = 8   #標準偏差
thetat = np.concatenate((np.reshape(np.array(b0), (1, )), b, np.reshape(np.array(sigma), (1, )))) 
X = np.reshape(uniform(-1.0, 5.0, n*p), (n, p))   #説明変数

##真のデータを発生
D = np.round(b0 + np.dot(X, b) + normal(0, sigma, n), 0)   #真の需要関数
S = np.round(b0 + np.dot(X, b) + uniform(0, 2.5, n), 0)   #真の供給関数

#購買データを発生(需要が供給を上回っている場合供給を購買データとする)
B = np.zeros((n))
for i in range(n):
    if D[i] > S[i]:
        B[i] = S[i]
    else:
        B[i] = D[i]

Data0 = np.concatenate((B[:, np.newaxis], D[:, np.newaxis], S[:, np.newaxis]), axis=1)   #データを結合
Data = pd.DataFrame(Data0, columns=["B", "D", "S"])


##打ち切りデータの指示変数を作成
z1 = np.array(Data.index[Data.D < Data.S]).astype(int)   #需要が満たされているデータ
z2 = np.array(Data.index[Data.D > Data.S]).astype(int)   #需要が供給を上回っているデータ


####打ち切りデータモデルを推定####
##対数尤度の定義
def define_likelihood(theta, D, B, X, z1, z2):
    
    #パラメータの設定
    beta0 = theta[0]
    beta1 = theta[range(1, theta.shape[0]-1)]
    sigma = theta[theta.shape[0]-1]
    
    #平均構造を設定
    Mu = beta0 + np.dot(X, beta1)

    #非打ち切りデータの対数尤度
    var = pow(sigma, 2)
    L1 = np.sum(-np.log(var) - pow((D - Mu)[z1], 2) / var)

    #打ち切りデータの対数尤度
    Lt = 1-norm.cdf(B - Mu, 0, sigma)[z2] 
    if sum(Lt==0)!=0:
        i = np.array(pd.DataFrame(Lt).index[Lt==0]).astype(int)
        Lt[i] = 10^-100
    L2 = np.sum(np.log(Lt))

    #対数尤度の和を取る
    LL = -np.sum(L1 + L2)
    return(LL)

##最尤法で打ち切りデータモデルのパラメータを推定
#パラメータの初期値を設定
X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
inv_XX = np.linalg.inv(np.dot(X1.T, X1))
XY = np.dot(X1.T, D)
betaf = np.dot(inv_XX, XY)
sigmaf = np.sqrt(np.var(D - np.dot(X1, betaf)))
theta = np.concatenate((betaf, np.reshape(np.array(sigmaf), (1, ))))   #初期パラメータを結合

#パラメータを最適化
res = optimize.minimize(define_likelihood, theta, args=(D, B, X, z1, z2), method='Powell')


####推定結果の確認と要約####
##推定されたパラメータと真のパラメータを比較
theta = res.x
LL = -res.fun
np.round(pd.DataFrame(np.hstack((np.reshape(theta, (theta.shape[0], 1)), np.reshape(thetat, (thetat.shape[0], 1))))), 3)


##適合度の確認
print(np.round(LL, 3))   #最大化された対数尤度
print(np.round(-2*LL + 2*theta.shape[0], 3))   #AIC
print(np.round(-2*LL + np.log(n)*theta.shape[0], 3))   #BIC


#####両側打ち切りモデル#####
####データの発生####
n = 10000   #サンプル数
p = 20   #説明変数数

##最大値が9以下、最小値が-4以上になるように変数と回帰係数を作成
T = 1000
for i in range(T):
    print(i)

    #パラメータの設定
    b = np.hstack((normal(0.24, 0.18, 12), normal(-0.21, 0.13, p-12)))   #回帰係数
    b0 = 0.6   #切片
    sigma = 0.5   #標準偏差

    #1～5の値をとる説明変数の発生
    X = np.round(np.reshape(normal(3, 1, n*p), (n, p)), 0)
    X[X > 5] = 5
    X[X < 1] = 1

    #真の応答変数の発生
    S = b0 + np.dot(X, b) + normal(0, 0.5, n)

    #応答変数が条件を満たしているならループを抜け出す
    if np.max(S) < 10 and np.max(S) > 7 and np.min(S) > -6 and np.min(S) < -2:
        break 

score_true = np.round(S, 0)   #真のスコア
thetat = np.concatenate((np.reshape(np.array(b0), (1, )), b, np.reshape(np.array(sigma), (1, )))) 
y = score_true

##データの確認と可視化
print(np.mean(S))
print(np.sqrt(np.var(S)))

plt.hist(S, bins=25)
plt.show()

##1以下および5以上のスコアにフラグを建て、観測データのスコアにする
#スコアの上限下限のフラグを建てる
upper_z = np.zeros((n))
lower_z = np.zeros((n))
upper_z[y > 5] = 1
lower_z[y < 1] = 1

#観測スコアの設定
y[y > 5] = 5
y[y < 1] = 1

#ヒストグラムの表示
plt.hist(y, bins=5)
plt.show()


####最尤法で両側打ち切りモデルを推定####
##両側打ち切りモデルの対数尤度関数
def define_likelihood(theta, y, X, lower_z, upper_z):

    #パラメータの設定
    beta0 = theta[0]
    beta1 = theta[range(1, theta.shape[0]-1)]
    sigma = theta[theta.shape[0]-1]
    z = abs(upper_z + lower_z - 1)
    
    #平均構造を設定
    Mu = beta0 + np.dot(X, beta1)

    #非打ち切りデータの対数尤度
    var = pow(sigma, 2)
    L1 = np.sum(-np.log(var) - pow((y[z==1] - Mu[z==1]), 2) / var)

    #上側打ち切りデータの対数尤度
    Lt1 = 1-norm.cdf(y[upper_z==1]  - Mu[upper_z==1] , 0, sigma)
    if sum(Lt1==0)!=0:
        i = np.array(pd.DataFrame(Lt1).index[Lt1==0]).astype(int)
        Lt1[i] = 10^-100
    L2 = np.sum(np.log(Lt1))
    
    #下側打ち切りデータの対数尤度
    Lt2 = norm.cdf(y[lower_z==1]  - Mu[lower_z==1], 0, sigma)
    if sum(Lt2==0)!=0:
        i = np.array(pd.DataFrame(Lt2).index[Lt2==0]).astype(int)
        Lt2[i] = 10^-100
    L3 = np.sum(np.log(Lt2))
    
    #対数尤度の和を取る
    LL = -np.sum(L1 + L2 + L3)
    return(LL)


##最尤法で打ち切りデータモデルのパラメータを推定
#パラメータの初期値を設定
X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
inv_XX = np.linalg.inv(np.dot(X1.T, X1))
XY = np.dot(X1.T, y)
betaf = np.dot(inv_XX, XY)
sigmaf = np.sqrt(np.var(D - np.dot(X1, betaf)))
theta = np.concatenate((betaf, np.reshape(np.array(sigmaf), (1, ))))   #初期パラメータを結合

#パラメータを最適化
res = optimize.minimize(define_likelihood, theta, args=(y, X, lower_z, upper_z), method='Nelder-Mead')


####推定結果の確認と要約####
##推定されたパラメータと真のパラメータを比較
theta = res.x
LL = -res.fun
np.round(pd.DataFrame(np.hstack((np.reshape(theta, (theta.shape[0], 1)), np.reshape(thetat, (thetat.shape[0], 1))))), 3)


##適合度の確認
print(np.round(LL, 3))   #最大化された対数尤度
print(np.round(-2*LL + 2*theta.shape[0], 3))   #AIC
print(np.round(-2*LL + np.log(n)*theta.shape[0], 3))   #BIC
6