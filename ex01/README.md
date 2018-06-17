## 线性问题

### 问题１(回归)

估计的一个连续性对象(回归)，假设噪声满足某一分布（泊松/高斯）

$y^i=\theta^Tx^i+\varepsilon^i$

$$ \varepsilon^i \backsim D$$

$$(y^i-\theta^Tx^i)\backsim D$$

我们的样本是独立重复采样获得。那么可以利用最大似然方法构造出样本作为独立事件的概率。

$$L(\theta)=\prod_{i=1}^{i=m}p(y^i|(x^i,\theta))$$

$$l(\theta)=\sum_{i=1}^{i=m}ln(p(y^i|(x^i,\theta)))$$

一般来说这个函数会是一个凸函数。我们可以使用的办法

#### 1 梯度上升法（感谢Newton，感谢Leibniz）

#### 2 牛顿切线（计算Hessian矩阵）

#### 3 解导数方程

限制最似然函数的最值确定$\theta$。

#### notice

关于$D$的处理我们有两种办法

##### 空间上D均匀

##### D是空间函数

以高斯分布为例，如果分布是均匀的($\frac{\partial D}{\partial x}=0$)，那么结果就是我们的常规似然函数。

如果分布不均匀，则$D(x)$,例如$\sigma $是$x$的函数

$$\sigma^2(x)=e^{-\frac{(x-x^i)^2}{2\tau^2}}$$

这样就可以突出局地的影响。

### 问题二（分类）

如果估计对象是一个二分类，即估计$x^i$具有性质$P$，对于一个样本我们做出$P/ \neg P$判断。

可以通过估计一个样本具有性质$P$的概率来求解这个问题。定义一个函数

$$h_{\theta}(x)=P(y\in P|(x,\theta))＝\frac{1}{1+e^{-\theta^Tx}}$$

那么

$$1-h_{\theta}(x)=P(y\notin P|(x,\theta))$$

$$p(y|(x,\theta))=(h_{\theta}(x))^{(y^i)}*(1-h_{\theta}(x))^{(y^i)}$$

利用最大似然方法也可以构造出似然函数。这个时候我们实际上是给出了样本空间($\Omega$)中每个位置可能具有性质$P$的概率分布。

### 问题三（多分类）

这个时候$P=(P_1,P_2,...,P_n)$是性质向量。如果我们认为分类是样本空间的划分$\forall x\in \Omega ,x\in P_i => x\notin P_j (j\ne i)$

这个时候我们相当于要给出样本空间中任意一个位置上属于每个分类的概率。

$$h_{\theta}(x)=[\frac{e^{\theta_1^Tx}}{\sum_{i=1}^{i=n}e^{\theta_i^Tx}},\frac{e^{\theta_2^Tx}}{\sum_{i=1}^{i=n}e^{\theta_i^Tx}},...,\frac{e^{\theta_n^Tx}}{\sum_{i=1}^{i=n}e^{\theta_i^Tx}}]$$

## 思考

前面一般都假设

$y|(x,\theta)\backsim D(x,\theta)$

例如二分类问题$y|(x,\theta)\backsim B(\phi(\theta,x)) $

对于回归问题我们假设$y|(x,\theta)\backsim N(\mu(\theta,x),\sigma^2) $($\sigma$固定)

扩大到一般的想法：

#### 1 假设$y|(x,\tau) \backsim D(\tau(x))$

#### 2 利用假设的分布求得似然函数$L(\tau(x))$，最大化似然函数得到$\tau(x)$的具体形式

其中$\tau(x)$是我们假设的模型，对于线性函数来说$\tau(x)=\theta^Tx$，也可以是其他奇奇怪怪的。

## $===>>>$我们的目标就是寻找到一个_适用于全样本空间的$\tau(x)$_，使得在我们的_分布假设_下观测是最容易发生的（最大似然的意义）。

###### 首先，这套思路是本身存在一定问题的。例如考虑回归问题，假设我们在样本上组建了一个回归假设$h(x)$,对于样本空间的任意一点$\forall x \in \Omega$ ,$\hat{y}=h(x)$是我们理论的回归值，$y$是理论上真实观测到的。误差$\varepsilon(x)=y-\hat{y}$,我们如何假设$\varepsilon(x)$的分布？正态？指数？泊松？$\chi^2$?

###### 其次，假如根据具体问题的经验选择了$N$,我们有什么证据表明误差在整个$\Omega$中都是同一类分布？会不会出现在一个区域上是$N$,另一个区域上是$\chi^2$?

###### 最后，我们的$\tau(x)$,选择什么形式的$\tau(x)$？）

如果不把关注的重心放在$\tau$的具体形式上，我们可以进一步推广广义线性模型。

## 关于广义线性模型

### 指数族分布

如果一个随机变量的概率密度函数形如

$f(y;(\eta,\kappa))=b(y)e^{\frac{ \eta^TT(y)-a(\eta)}{c(\kappa)}}$

那么这个分布就叫作广义指数族分布。这个模型其实是对一般常见分布的一种泛化。

对于$y \backsim Bernoulli(n,\phi)=>P(m)=\binom n m  \phi^m(1-\phi)^{n-m}(m\in{0,1,2,...,n})$

$f(m)=P(m)=\binom n m e^{m\ln\phi+(n-m)\ln(1-\phi)}=\binom n m e^{m\ln\frac{\phi}{1-\phi}+n\ln(1-\phi)}$

$\eta=\ln(\frac{\phi}{1-\phi})$

$T(y)=y$

$a(\eta)=-nln(1-\phi)$

$c(\eta)=1$

$y\backsim P(\lambda)=>P(y=k)=\frac{e^{-\lambda }\lambda^k}{k!}$

$f(k)=P(k)=\frac{e^{-\lambda }}{k!}\lambda^k=\frac{e^{-\lambda }}{k!}e^{k\ln\lambda}$显然也是指数族。

$y\backsim E(\lambda)=>f(y)=\lambda e^{-\lambda y}(y>0)$

 $b(y)=\lambda\{y>0\}$

............

### 指数族分布性质

$E(T(y))=\frac{\partial a(\eta)}{\partial \eta}$

$var(T(y))=c(\kappa) \frac{\part^2 a(\eta)}{\part \eta^2}\ge0$半正定

证明：

$\int_{-\infty}^{+\infty}b(y)e^{\frac{ \eta^TT(y)-a(\eta)}{c(\kappa)}}dy=1$

$\int_{-\infty}^{+\infty}b(y)e^{\frac{ \eta^TT(y)}{c(\kappa)}}dy=e^{\frac{a(\eta)}{c(\kappa)}}$

$\frac{\part \int_{-\infty}^{+\infty}b(y)e^{\frac{ \eta^TT(y)}{c(\kappa)}}dy}{\part \eta}= \int_{-\infty}^{+\infty}b(y)e^{\frac{ \eta^TT(y)}{c(\kappa)}}\frac{T(y)}{c(\kappa)} dy=e^{\frac{a(\eta)}{c(\kappa)}}\frac{\frac{\part a(\eta)}{\part \eta} }{c(\kappa)}$

$=> \int_{-\infty}^{+\infty}\frac{T(y)}{c(\kappa)}  b(y)e^{\frac{ \eta^TT(y)-a(\eta)}{c(\kappa)}}dy=\frac{\frac{\part a(\eta)}{\part \eta} }{c(\kappa)}$

$=> \int_{-\infty}^{+\infty}T(y)  b(y)e^{\frac{ \eta^TT(y)-a(\eta)}{c(\kappa)}}dy=\frac{\part a(\eta)}{\part \eta} $

$=>E(T(y))=\frac{\part a(\eta)}{\part \eta}$

$var(T(y))=E(T^2(y))-E(T(y))^2$

$ \int_{-\infty}^{+\infty}T(y)  b(y)e^{\frac{ \eta^TT(y)}{c(\kappa)}}dy=e^{\frac{a(\eta)}{c(\kappa)}}\frac{\part a(\eta)}{\part \eta} $

 $\int_{-\infty}^{+\infty}T^2(y)  b(y)e^{\frac{ \eta^TT(y)}{c(\kappa)}}dy=c(\kappa)e^{\frac{a(\eta)}{c(\kappa)}}\frac{\part^2 a(\eta)}{\part \eta^2}+c(\kappa)e^{\frac{a(\eta)}{c(\kappa)}}(\frac{\part a(\eta)}{\part \eta})^2$ 

$E(T^2(y))=\int_{-\infty}^{+\infty}T^2(y)  b(y)e^{\frac{ \eta^TT(y)-a(\eta)}{c(\kappa)}}dy=c(\kappa)\frac{\part^2 a(\eta)}{\part \eta^2}+(\frac{\part a(\eta)}{\part \eta})^2$

$var(T(y))=c(\kappa)\frac{\part^2 a(\eta)}{\part \eta^2}$

### 利用任意的一个指数族分布构造似然函数

这个时候不关心具体的$\tau$是一种什么形式。我们通过$\tau$的模型预报$y$$=>y=\tau(x)$,现我们观测到样本$\{(x^1,y^1),(x^2,y^2),...,(x^m,y^m)\}$，更一般地，我们假设$y^i|(x^i;\tau)\backsim D(x^i,\tau)$,$D$为任意一个分布。按照最大似然的原理：

$\mathop{argmax }_{\tau}  L(\tau)=\prod_{i=1}^{i=m}d(y^i|(x^i;\tau))  \leftrightarrow \mathop{argmax }_{\tau}  l(\tau)=\sum_{i=1}^{i=m}\ln d(y^i|(x^i;\tau))  $

如果$D$是一个常见的指数族分布，$y^i|(x^i;\tau)\backsim D(\eta(x^i,\tau),\kappa(x^i,\tau))$

$l(\tau)=\sum_{i=1}^{i=m}\ln d(y^i|(x^i;\tau)) =\sum_{i=1}^{i=m}\ln b(y^i)e^{\frac{ \eta^TT(y^i)-a(\eta)}{c(\kappa)}} =\sum_{i=1}^{i=m}\ln b(y^i)+\sum_{i=1}^{i=m}\frac{ \eta^TT(y^i)-a(\eta)}{c(\kappa)}$

对于我们假定的一个指数分布$\sum_{i=1}^{i=m}\ln b(y^i)=const$,$=>\mathop{argmax }_{\tau} \sum_{i=1}^{i=m}\frac{ \eta^TT(y^i)-a(\eta)}{c(\kappa)}$

对于具体的一个样本而言$\frac{\part \frac{\eta^TT(y^i)-a(\eta)}{c(\kappa)}}{\part \eta}=\frac{T(y^i)-a'(\eta)}{c(\kappa)}=0=>T(y^i)=a'(\eta)=E(T(y^i))$

$\frac{\part^2 \frac{\eta^TT(y^i)-a(\eta)}{c(\kappa)}}{\part \eta^2}\ge0$半正定，所以调整参数$\eta$使得$T(y)=E(T(y))=a'(\eta(x,\tau))$就是最大似然函数的最佳情况。

####===>>>结论：对于选定的一个指数族分布只要保证对于任给的一个样本能够调整$\tau$使得$T(y)=E(T(y))=a'(\eta(x,\tau))$,就完成了理想化的最大似然估计。但是现实中可能样本数量过于巨大,而$\tau$可调整参数较少，使得上述等式不一定都能满足。

### $\tau$的具体形式

注意上述推导中我们都是假设$\eta$可以是一个向量进行的。我们实际进行的是一个假设$h:R^n\rightarrow R^k$的寻找过程。对于线性模型，$\eta=\tau(x)=\theta^Tx$，其中$\theta$是一个$n\times k$的矩阵($softmax$),满足$T(y)=a'(\theta^Tx)$,这样对于样本$\{(x^1,y^1),(x^2,y^2),...,(x^m,y^m)\}$，使得$T(y^i)=a'(\theta^Tx^i) ,i\in{1,2,...,m}$可能无法办到(例如样本数量$m\gt nk$,对于线性模型，上述等式极可能不都成立。)

但是我们可以退回到最大似然估计$\mathop{argmax }_{\tau} \sum_{i=1}^{i=m} \eta^TT(y^i)-a(\eta)$ ,再次感谢$Newton$,梯度上升。

其他的，改进$\eta=\tau(x)=\theta^Tx=>\eta_i=\theta_i^Tx,i\in{1,2,...,k}$

当然我们也可以$\eta_i=x^TAx+Bx+C,i\in{1,2,...,k}$

### $softmax$说明具体的操作过程

对于多分类问题，假设存在$k$个类，$\Omega=\Omega_1+\Omega_2,\Omega_3+...+\Omega_k,\Omega_i \cap\Omega_j=\emptyset(i\ne j)$

假设$p(y\in \Omega_i|(x,\tau))=\phi_i$,那么$\sum_{i=1}^{i=k}\phi_i=1$

$p(y|(x,\tau))=(1-\sum_{i=1}^{i=k-1}\phi_i)^{is(y\in\Omega_k)}\prod_{i=1}^{i=k-1}\phi_i^{is(y\in\Omega_i)}$

化为标准的指数族分布：$p(y|(x,\tau))=e^{\ln(1-\sum_{i=1}^{i=k-1}\phi_i)^{is(y\in\Omega_k)}+\sum_{i=1}^{i=k-1}\ln\phi_i^{is(y\in\Omega_i)}}$

显然$c(\kappa)=1,b(y)=1$

${\ln(1-\sum_{i=1}^{i=k-1}\phi_i)^{is(y\in\Omega_k)}+\sum_{i=1}^{i=k-1}\ln\phi_i^{is(y\in\Omega_i)}}=(\ln(\frac{\phi_1}{\phi_k}),\ln(\frac{\phi_2}{\phi_k}),...,\ln(\frac{\phi_{k-1}}{\phi_k})) \cdot T(y) +\ln(\phi_k)$

$T(y)=(is(y\in \Omega_1),is(y\in \Omega_2),...,is(y\in \Omega_{k-1}))^T$

$\eta^T=(\ln(\frac{\phi_1}{\phi_k}),\ln(\frac{\phi_2}{\phi_k}),...,\ln(\frac{\phi_{k-1}}{\phi_k}))$

$\eta=\tau(x)$所以要寻找$\tau(x)$和$\phi_i$的关系需要反解出$\phi_i$

$\eta_i=\ln(\frac{\phi_i}{\phi_k}),i\in{1,2,...,k} => \phi_i=\phi_ke^{\eta_i}$

$1=\sum_{i=1}^{i=k}\phi_i=\phi_k\sum_{i=1}^{i=k}e^{\eta_i}=>\phi_k=\frac{1}{\sum_{i=1}^{i=k}e^{\eta_i}}$对于线性的模型$\eta^T=\theta^Tx=(\theta_1^Tx,...,\theta_k^Tx)$

$\phi_i=\frac{e^{\eta_i}}{\sum_{j=1}^{j=k}e^{\eta_j}}=\frac{e^{\theta_i^Tx}}{\sum_{j=1}^{j=k}e^{\theta_j^Tx}}$

$l(\tau)=\sum_{i=1}^{i=m} (\eta^T(x^i,\tau)T(y^i)+\phi_{k})=\sum_{i=1}^{i=m}((\theta_1^Tx^i,...,\theta_{k-1}^Tx^i)\cdot T(y)＋\ln(\phi_{k}))=\sum_{i=1}^{i=m}(\theta_j^Tx^i)^{y\in \Omega_j}=\sum_{i=1}^{i=m}(\frac{e^{\theta_j^Tx}}{\sum_{l=1}^{l=k}e^{\theta_l^Tx}})^{y\in \Omega_j}$

再利用梯度上升算法。













