#####ポアソン回帰モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib
from scipy.special import gammaln
from scipy.misc import factorial
from numpy.random import *
from scipy import optimize


####データの発生####
#np.random.seed(142341)   #シードを作成
#データの設定
k = 11   #パラメータ数
N = 3000   #サンプル数


##説明変数の発生
#連続変数の発生
cont1 = 3
intercept = np.transpose(np.matlib.repmat(1, 1, N))
X_cont = randn(N, cont1)

#二値変数の発生
bin1 = 3
X_bin = np.zeros((N, bin1))

for i in range(X_bin.shape[1]):
    p_bin = (0.7 - 0.4) * rand() + 0.4
    X_bin[:, i] = binomial(1, p_bin, N)
    
#多値変数の発生
multi1 = 4
p_multi = np.array((0.2, 0.2, 0.3, 0.3))
X_multi = multinomial(1, p_multi, N)
multi_sums = np.sum(X_multi, axis=0)
X_multi = np.delete(X_multi, multi_sums.argmin(), axis=1)   #冗長な変数の削除

#データの結合
X = np.concatenate((intercept, X_cont, X_bin, X_multi), axis=1)
pd.DataFrame(X)


##応答変数の発生
#パラメータの設定
beta00 = uniform(0.5, 0.7, 1)
beta01 = uniform(0, 0.4, cont1)
beta02 = uniform(-0.4, 0.6, bin1)
beta03 = uniform(-0.5, 0.8, multi1-1)
beta0 = np.concatenate((beta00, beta01, beta02, beta03))

#ポアソンモデルのパラメータを計算し、ポアソン分布から応答変数を発生
lam = np.exp(np.dot(X, beta0))
y = poisson(lam, N)
plt.hist(y, bins=15)
plt.show()


####最尤法でポアソン回帰モデルを推定####
##ポアソン回帰モデルの対数尤度関数
def likelihood(beta, y, X):

    #平均構造の定義
    lam = np.exp(np.dot(X, beta))
    log_lambda = np.log(lam)   #リンク関数

    #対数尤度を計算
    LLi = y*np.log(lam)-lam - gammaln(y+1)
    LL = -np.sum(LLi)
    return LL

##準ニュートン法で対数尤度を最大化
theta0 = np.zeros(X.shape[1])   #初期値の設定
res = optimize.minimize(likelihood, theta0, args=(y, X), method='Powell')

#推定されたパラメータと真のパラメータの比較
print(pd.DataFrame(np.round(np.array((res.x, beta0)), 3)))
np.round(-res.fun, 3)   #最大化された対数尤度

