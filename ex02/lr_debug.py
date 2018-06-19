from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):#为属性添加常数列
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):#加载数据设置X,Y
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    #线性化
    #X=X/(np.max(X,0)-np.min(X,0))
    #施加高斯噪声
    #X=X+0.0001*np.random.standard_normal(np.array(X).shape)
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)  
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))
    return grad#+1e-6*theta

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)#参数的个数
    #随机化theta
    theta=np.random.randn(n)
    learning_rate = 10#学习的效率

    i = 0
    error=np.zeros(100)
    while  i!=1000000-1:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        #learning_rate=learning_rate*1.0/i**2
        theta = theta  - learning_rate * (grad)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print('grad norm2  :%10.9f '%np.linalg.norm(grad))
            print(prev_theta - theta)
            error[int(i/10000)]=np.linalg.norm(prev_theta - theta)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    print(theta)
    return error

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('data_a.txt')
   # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('data_b.txt')
    error=logistic_regression(Xb, Yb)
    pl.plot(range(1,101),error)
    pl.show()
    return

if __name__ == '__main__':
    main()
