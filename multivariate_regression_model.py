#####多変量回帰モデル#####
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import numpy.matlib
import scipy.linalg
from numpy.random import *
from scipy import optimize
import seaborn as sns


####任意の相関行列(分散共分散行列)を作成する関数####
##任意の相関行列を作る関数
def CorM(col, lower, upper, eigen_lower, eigen_upper):
    #相関行列の初期値を定義する
    cov_vec = (upper - lower) *rand(col*col) + lower   #相関係数の乱数ベクトルを作成
    rho = np.reshape(np.array(cov_vec), (col, col)) * np.tri(col)   #乱数ベクトルを下三角行列化
    Sigma = np.diag(np.diag(rho + rho.T) + 1) - (rho + rho.T)   #対角成分を1にする
    
    #相関行列を正定値行列に変更
    #固有値分解を実行
    eigen = scipy.linalg.eigh(Sigma)
    eigen_val = eigen[0] 
    eigen_vec = eigen[1]
    
    #固有値が負の数値を正にする
    for i in range(eigen_val.shape[0]-1):
        if eigen_val[i] < 0:
            eigen_val[i] = (eigen_upper - eigen_lower) * rand(1) + eigen_lower
            
    #新しい相関行列の定義と対角成分を1にする
    Sigma = np.dot(np.dot(eigen_vec, np.diag(eigen_val)), eigen_vec.T)
    normalization_factor = np.dot(pow(np.diag(Sigma), 0.5)[:, np.newaxis], pow(np.diag(Sigma), 0.5)[np.newaxis, :])
    Cor = Sigma / normalization_factor
    return Cor


##相関行列から分散共分散行列に変換する関数
def covmatrix(Cor, sigma_lower, sigma_upper):
    sigma = (sigma_upper - sigma_lower) * rand(np.diag(Cor).shape[0]) + sigma_lower
    sigma_factor = np.dot(sigma[:, np.newaxis], sigma[np.newaxis, :])
    Cov = Cor * sigma_factor
    return Cov

##任意の分散共分散行列を発生させる
#パラメータを設定
col = 10
lower = -0.9
upper = 0.9
eigen_lower = 0
eigen_upper = 0.1
sigma_lower = 1.5
sigma_upper = 2.0

#相関行列を発生させる
Cor = CorM(col=col, lower=lower, upper=upper, eigen_lower=eigen_lower, eigen_upper=eigen_upper)
print(scipy.linalg.eigh(Cor)[0])   #正定値かどうか確認
print(np.round(Cor, 3))   #相関行列を確認

#分散共分散行列に変換
Cov = covmatrix(Cor=Cor, sigma_lower=sigma_lower, sigma_upper=sigma_upper)
print(scipy.linalg.eigh(Cov)[0])
print(np.round(Cov, 3))

##多変量正規分布から乱数を発生させる
X = np.zeros((1000, col))
mu = np.zeros(col)

for i in range(X.shape[0]):
    X[i, ] = np.random.multivariate_normal(mu, Cor)

print(np.round(np.corrcoef(X.transpose()), 3))   #発生させた相関行列
print(np.round(Cor, 3))   #真の相関行列


####データの発生####
#データの設定
n = 2000   #サンプル数
k = 6   #応答変数数


##説明変数の発生
#連続変数の発生
cont = 4
X_cont = randn(n, cont)

#二値変数の発生
bin = 3
X_bin = np.zeros((n, bin))

for i in range(X_bin.shape[1]):
    p_bin = (0.7 - 0.4) * rand() + 0.4
    X_bin[:, i] = binomial(1, p_bin, n)

#多値変数の発生
multi = 4
p_multi = np.array((0.2, 0.2, 0.3, 0.3))
X_multi = multinomial(1, p_multi, n)

multi_sums = np.sum(X_multi, axis=0)
print(multi_sums)
X_multi = np.delete(X_multi, multi_sums.argmin(), axis=1)   #冗長な変数を削除

#説明変数を結合
intercept = np.ones((n, 1))
X = np.concatenate((intercept, X_cont, X_bin, X_multi), axis=1)


##応答変数の発生
#回帰パラメータの発生
BETA0 = np.zeros([X.shape[1], k])

for i in range(BETA0.shape[1]):
    beta01 = uniform(1.5, 3.0, 1)
    beta02 = uniform(0, 1.0, cont)
    beta03 = uniform(-1.2, 1.6, (bin+multi-1))
    BETA0[:, i] = np.concatenate((beta01, beta02, beta03), axis=0).T

#分散共分散行列の発生
lower = -0.75
upper = 0.9
eigen_lower = 0
eigen_upper = 0.1
sigma_lower = 1.5
sigma_upper = 2.2

#相関行列を発生させる
Cor0 = CorM(col=k, lower=lower, upper=upper, eigen_lower=eigen_lower, eigen_upper=eigen_upper)
print(np.round(Cor, 3))   #相関行列を確認

#分散共分散行列に変換
Cov0 = covmatrix(Cor=Cor0, sigma_lower=sigma_lower, sigma_upper=sigma_upper)
print(np.round(Cov, 3))


#回帰モデルの平均構造を設定
Z = np.dot(X, BETA0)

#平均構造に多変量正規分布の誤差を加える
Y = np.zeros((n, k))

for i in range(Y.shape[0]):
    Y[i, ] = np.random.multivariate_normal(Z[i, :], Cov0)

#発生させた応答変数の集計
print(np.average(Y, axis=1))
print(np.corrcoef(Y))

#scatter_matrix(pd.DataFrame(Y), diagonal="kde", color="k", alpha=0.3)
Y_pd = pd.DataFrame(Y)
scatter_matrix(Y_pd, diagonal='kde', color='k', alpha=0.3)
plt.show()


####最小二乗法で多変量回帰モデルを推定####
##最小二乗法でbetaを推定
inv_XX = np.linalg.inv(np.dot(X.T, X))
XY = np.dot(X.T, Y)
BETA = np.dot(inv_XX, XY)

#推定結果と真の値の比較
print(pd.DataFrame(np.round(BETA, 3)))
print(pd.DataFrame(np.round(BETA0, 3)))


##分散共分散行列を計算
XB = np.dot(X, BETA)   #予測値
error = Y - XB   #誤差
Cov = np.dot(error.T, error) / Y.shape[0]   #分散共分散行列

#推定結果と真の値の比較
print(pd.DataFrame(np.round(Cov, 3)))
print(pd.DataFrame(np.round(Cov0, 3)))

#相関行列に変換
normalization_factor = np.dot(pow(np.diag(Cov), 0.5)[:, np.newaxis], pow(np.diag(Cov), 0.5)[np.newaxis, :])
Cor = Cov / normalization_factor

normalization_factor = np.dot(pow(np.diag(Cov0), 0.5)[:, np.newaxis], pow(np.diag(Cov0), 0.5)[np.newaxis, :])
Cor0 = Cov0 / normalization_factor

#相関行列と真の値の比較
print(pd.DataFrame(np.round(Cor, 3)))
print(pd.DataFrame(np.round(Cor0, 3)))

####適合度の計算#####
##AICを計算
#多変量正規分布の対数尤度を計算する関数

