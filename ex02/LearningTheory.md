## 学习理论

####$Markov$不等式和$Chernoff$边界

对随机变量 $X$ 和 $\varepsilon>0$ , 有：

$p(|x|\ge\varepsilon)\le \frac{1}{\varepsilon^\alpha}E(|x|^\alpha),\forall \alpha>0$

证明:

$ \frac{1}{\varepsilon^\alpha}E(|x|^\alpha)=\int_{\Omega}\frac{|x|^\alpha}{\varepsilon^\alpha}f(x)dx\ge \int_{|x|\ge\varepsilon}\frac{|x|^\alpha}{\varepsilon^\alpha}f(x)dx \ge \int_{|x|\ge\varepsilon}f(x)dx =p(|x|\ge \varepsilon)$

当$\alpha=2$叫$Chebyshev$不等式。



$Chernoff$边界：

假设$X=\sum_{i=1}^{i=n}X_i,X_i$是独立泊松实验(每次的概率分布不一定相同)，$\mu=E(X)$,$\forall \delta>0$

$P(X\ge(1+\delta)\mu)\le(\frac{e^\delta}{(1+\delta)^{1+\delta}})^\mu$

记$t=\ln(1+\delta)>0$

$X\ge(1+\delta)\mu) \leftrightarrow e^{tX}\ge e^{t(1+\delta)\mu}$

$P(e^{tX}\ge e^{t(1+\delta)\mu})\le \frac{E(e^{tX})}{e^{t(1+\delta)\mu}}$(Markov)

$E(e^{tX})=E(e^{t\sum_{i=1}^{i=n}X_i})=E(\prod_{i=1}^{i=n}e^{tX_i})=\prod_{i=1}^{i=n}E(e^{tX_i})$

$P(X_i=1)=p_i$,$\mu=E(X)=\sum_{i=1}^{i=n}E(X_i)=\sum_{i=1}^{i=n}p_i$

$E(e^{tX_i})=e^{t}p_i+e^{0}(1-p_i)=p_i(e^t-1)+1\le e^{p_i(e^t-1)}$

$E(e^{tX})\le e^{\mu(e^t-1)}$

$P(X\ge(1+\delta)\mu)=P(e^{tX}\ge e^{t(1+\delta)\mu})\le \frac{E(e^{tX})}{e^{t(1+\delta)\mu}}\le \frac{e^{\mu(e^t-1)}}{e^{t(1+\delta)\mu}}=(\frac{e^{(e^t-1)}}{e^{t(1+\delta)}})^\mu=(\frac{e^\delta}{(1+\delta)^{1+\delta}})^\mu$

对于$0<\delta<1$,$(1+\delta)\ln(1+\delta)\ge\delta- \frac{\delta^2}{2}$

$f(\delta)=(1+\delta)\ln(1+\delta)-\delta+ \frac{\delta^2}{2}$

$f(0)=0$

$f^{'}(\delta)=ln(1+\delta)+1-1+\delta=ln(1+\delta)+\delta>0$

$P(X\ge(1+\delta)\mu)=(\frac{e^\delta}{(1+\delta)^{1+\delta}})^\mu= (\frac{e^\delta}{e^{(1+\delta)\ln(1+\delta)}})^\mu \le e^{-\frac{\mu\delta^2}{2}} $

$P(|X-\mu|\ge \delta\mu)\le 2e^{-\frac{\mu\delta^2}{2}}$

对文档上的情况$X=\frac{1}{m}\sum_{i=1}^{i=n}X_i$

$P(|X-\mu|\ge\delta) \le 2e^{-2m\delta^2}$

### 1 finite H

$H=\{h_i,h_2,..,h_k\}$

$\hat{\varepsilon}_{h_i}=\frac{1}{m}\sum_{i=1}^{i=m}1\{h_i(x^i)\ne y^i\}$

其实接下来的思路就是类似于假设检验。

假设样本$\{(x^1,y^1),...,(x^m,y^m)\}$来自一个$Bernoulli$分布。记$Z_j=1\{h_i(x^j)\ne y^j)\}$

$\hat{\varepsilon}(h_i)=\frac{1}{m}\sum_{j=1}^{j=m}1\{h_i(x^j)\ne y^j\}=\frac{1}{m}\sum_{j=1}^{j=m}Z_j$

$E(\hat{\varepsilon}(h_i))=\frac{1}{m}\sum_{j=1}^{j=m}E(Z_j)=E(Z)=\varepsilon(h_i)$

有$Chernoff$边界：

$P(|\hat{\varepsilon}(h_i)-\varepsilon(h_i)|\ge\delta)\le 2e^{-2m\delta^2}$

这样在大样本的情况下，我们有信心觉得训练误差和泛化误差足够接近(其实就是大数定理,当我们样本足够大的时候样本平均值会一致趋近于分布平均值).

对于$H$整体的误差$,P(\exist h_i\in H| |\hat{\varepsilon}(h_i)-\varepsilon(h_i)|\ge\delta)\le 2ke^{-2m\delta^2}$

##### 样本中得出的最优解和实际$H$中最优解的误差估计很有意思

$\varepsilon(\hat{h})\le(\min \limits_{h\in H}\varepsilon(h))+2\sqrt{\frac{1}{2m}log\frac{2k}{\delta}}$

也就是我们估计出来的模型，与理论上最好的模型的泛化误差在$1-\delta$的可信性下不超过$2\sqrt{\frac{1}{2m}log\frac{2k}{\delta}}$

(如果$k\to \infty$)误差估计不大令人满意。