#####ロジスティック回帰モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
from numpy.random import *
from scipy import optimize


####データの発生####
#np.random.seed(142341)   #シードを作成
#データの設定
k = 11   #パラメータ数
N = 3000   #サンプル数

#パラメータの設定
beta1 = rand(k)

#説明変数の発生
X = randn(N, k-1)
intercept = np.transpose(np.matlib.repmat(1, 1, N))
XM = np.concatenate((intercept, X), axis=1)


#ロジットと確率の計算
logit = np.dot(XM, beta1)   #ロジットの計算
Pr = np.exp(logit) / (1+np.exp(logit))   #確率の計算

#ベルヌーイ乱数から応答変数の発生
y = binomial(1, Pr) 
np.mean(y)


####最尤法でロジスティック回帰モデルを推定####
##ロジスティック回帰モデルの対数尤度を定義
def define_likelihood(beta, y, X):
    #ロジットと確率の定義
    logit = np.dot(X, beta)
    Pr = np.exp(logit)/(1+np.exp(logit))
    
    #対数尤度の計算
    LLi = y * np.log(Pr) + (1-y) * np.log(1-Pr)
    LL = -sum(LLi)   
    return LL


##対数尤度を最大化する
theta0 = np.zeros(k)   #初期値の設定
res = optimize.minimize(define_likelihood, theta0, args=(y, XM), method='Powell')

np.round(res.x, 3)   #推定されたパラメータ
np.round(beta1, 3)   #真のパラメータ


