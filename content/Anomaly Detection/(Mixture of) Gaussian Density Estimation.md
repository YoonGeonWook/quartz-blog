---
sticker: emoji//1f4d5
tags:
  - Anomaly_Detection
  - 이상탐지
  - density-based
  - Gaussian
  - Mixture_of_Gaussian
banner: Anomaly Detection/image/gaussian.png
---
# 🎲 Density-based Novelty Detection

## Purpose
- 이상치 탐지의 Abnormal data에 대한 두 번째 정의에 중점을 둔 방식
	- 주어진 데이터를 활용해서 정상인 Normal data가 가질 수 있는 분포를 먼저 추정한 다음에 그 추정된 분포를 통해서 새로운 객체가 들어왔을 때 그 객체가 발생할 확률이 높으면 Normal data로 판별하고 그렇지 않을 경우 Abnormal data로 판별하는 것이다.
- [x] Estimate the data-driven density function
- [x] If a new instance has a **low probability** according the trained density function, it will be identified as novel.
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112101704.png)
Figure 1 : 1차원 데이터에 대해서 위와 같이 히스토그램이 그려진다고 하자. 실질적으로는 분포가 Gaussian 인지 아닌지 알 수 없음에도 불구하고 Gaussian distribution을 가정을 하고, 노란색으로 표시된 것과 같은 정규분포를 추정할 수 있다. 이렇게 분포를 추정한 뒤에 새로운/test 객체가 파란색과 같이 분포가 가질 수 있는 확률이 높은 경우에는 Normal로 판단되는 반면 빨간색과 같은 곳의 경우 Normal 데이터로부터 생성될 확률이 상당히 낮기 때문에 Abnormal 로 판정한다.

![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112102715.png)

- Kernel Density Estimation : Parzen Window Density Estimation
	- 각각의 객체는 모두 Gaussian distribution의 중심임을 가정하고, 그로부터 주어진 정상 데이터 영역의 밀도 함수를 추정하겠다는 것 

# 🎲 Gaussian Density Estimation

- **Assume** that the observed data are drawn from a Gaussian distribution
	- 실제가 아니라 가정을 하는 것임!
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112103126.png)

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]
$$
- 찾아야할 미지수 (모수) : 
	- 여기서 $\mathbf{x}_i\in X^+$ 는 Normal data의 집합 $X^+$의 원소들만을 사용하겠다는 것
	- mean vector : $\boldsymbol{\mu}=\frac{1}{n}\sum_{\mathbf{x}_i\in\mathbf{X}^+}\mathbf{x}_i$
	- covariance matrix : $\Sigma=\frac{1}{n}\sum_{\mathbf{x}_i\in\mathbf{X}^+}(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T$ 
	- 결론적으로 normal data $X^+$가 주어졌을 때 normal data의 평균과 공분산을 계산해야 하는 것이다


#### Advantages : 
1) **Insensitive to scaling of the data**
	- 다음과 같은 데이터가 있다고 해보자 : $$\begin{array}{cccc} & V_1 & V_2 & V_3 \\X_1 & 1.0 & 1000 & 0.01 \\X_2 & 1.1 & 980 & 0.02 \\\vdots & \vdots & \vdots & \vdots \\X_n & 0.9 & 1020 & 0.05\end{array}$$
	- Covariance matrix의 역행렬을 이용하기 때문에 변수별로 단위가 다른 것이 영향을 미치지 않음 = insensitive / robust
	- 즉, 변수에 대한 normalization을 하지 않아도됨
2) **Possible to compute analyrically the optimal threshold**
	- rejection에 대한 제 1종오류를 먼저 정의할 수 있다. 예를 들어 신뢰수준 95%까지만 포함하겠다 라고 하면 처음부터 주어진 데이터 기준으로 5% 정도는 rejection이 된다는 것, 실제로는 정상 데이터이지만 abnormal로 reject 되는 것음 감수하고 boundary/cut-off를 계산한다는 것임
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112104053.png)

#### Maximum Likelihood Estimation ; MLE
- 1차원 데이터인 경우, 실질적으로 추정해야 할 미지수/모수 : 
	- Parameter : $\mu$ and $\sigma^2$
	$$\begin{aligned}L&=\prod_{i=1}^NP(x_i|\mu,\sigma^2)=\prod_{i=1}^N\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)\\\log L&=-\frac{1}{2}\sum_{i=1}^N\frac{(x_i-\mu)^2}{\sigma^2}-\frac{N}{2}\log(2\pi \sigma^2)\end{aligned}$$
	- Let's set $\gamma = 1/\sigma^2$ : 
	$$\begin{aligned}\log L&=-\frac{1}{2}\sum_{i=1}^N\gamma(x_i-\mu)^2-\frac{N}{2}\log(2\pi)+\frac{N}{2}\log(\gamma) \\\frac{\partial\log L}{\partial\mu}&=\gamma\sum_{i=1}^N(x_i-\mu)=0\to \mu=\frac{1}{N}\sum_{i=1}^Nx_i\\\frac{\partial\log L}{\partial\gamma}&=-\frac{1}{2}\sum_{i=1}^N(x_i-\mu)^2+\frac{N}{2\gamma}=0\to\sigma^2=\frac{1}{N}\sum_{i=1}^N(x_i-\mu)^2\end{aligned}$$
		- **$\mu$ 의 추정치는 Train data의 normal들의 평균, $\sigma^2$의 추정치는 normal들의 분산**
- In general, (multivariate case) : 
$$\boldsymbol{\mu}=\frac{1}{N}\sum_{i=1}^N\mathbf{x}_i,\qquad \boldsymbol{\Sigma}=\frac{1}{N}\sum_{i=1}^N(\mathbf{x}_i-\boldsymbol{\mu})(\mathbf{x}_i-\boldsymbol{\mu})^T$$
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112110946.png)

- 이제 여기에서 한 가지 이슈가 있는데, 실질적으로 공분산 행렬을 변수의 개수만큼의 차원을 갖는 squared matrix 이기 때문에 이를 어떻게 처리하느냐에 따라서 추정되는 분포의 모양 (shape)이 약간씩 달라지게 된다. 
- The shape of Gaussian distribution according to the Covariance matrix type
	- **Spherical** : $$\Sigma=\sigma^2\left[\begin{array}{ccc}1 & \cdots & 0 \\\vdots & \ddots & \vdots \\0 & \cdots & 1\end{array}\right]$$
		- 모든 변수가 동일한 분산을 가지고 있다는 가정임. 이는 변수들간 독립을 가정하고 있음
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112111249.png)

- **Diagonal** : $$\Sigma=\left[\begin{array}{ccc}\sigma_1^2 & \cdots & 0 \\\vdots & \ddots & \vdots \\0 & \cdots & \sigma_d^2\end{array}\right]$$
	- 여전히 독립이지만 변수별로 다른 분산을 가질 수 있다
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112111456.png)
- **Full** : $$\Sigma=\left[\begin{array}{ccc}\sigma_{11} & \cdots & \sigma_{1 d} \\\vdots & \ddots & \vdots \\\sigma_{d 1} & \cdots & \sigma_{d d}\end{array}\right]$$
	- 더 이상 변수들 간 독립이 아닌 어느정도의 상관관계를 가짐
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112111545.png)

여기까지가 전체 데이터를 하나의 Gaussian 분포로 가정하고, 그로부터 abnormal score를 계산하는 방법이다. 이때 abnormal score라 함은 추정된 pdf로 부터 새로운 데이터가 들어왔을 때 산출되는 확률분포의 값이다. 그렇기 때문에 그 값이 낮을수록 abnormal일 확률이 커지고, 그 값이 높을수록 abnormal일 확률은 낮아진다. 
- Anomaly Score = 1 - p(x)
# 🎲 Mixture of Gaussian Density Estimation

- Mixture of Gaussian (MoG) Density Estimation : 
	- Gaussian Density Estimation - assumes **a very strong model** of the data : <font style="color:blue">unimodal and convex</font>
	- MoG : 
		- an extension of Gaussian that allows <font style="color:blue">multi-modal</font> distribution
		- <font style="color:blue">a linear combination of normal distributions</font>
		- Has a smaller bias than the single Gaussian distribution, but requires far more data for training
			- unimomal의 경우에는 평균과 분산 2개만 계산하면 되었지만, MoG에서는 각 가우시안에 대해서 $w_i,\mu_i,\sigma_i$ 를 추정해야 되므로 K개의 Gaussian을 사용한다면 $3\times K$개를 계산해야 하고, 적절한 $K$에 대해 train data에서 탐색해야 함
$$
f(x) = w_1\cdot \mathcal{N}(\mu_1,\sigma_1^2) + w_2\cdot \mathcal{N}(\mu_2,\sigma_2^2) + w_3\cdot \mathcal{N}(\mu_3,\sigma_3^2)
$$
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112112338.png)

#### MoG example

![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112112732.png)

#### Components of MoG : 
- 어떤 객체가 normal class에 속할 확률 : $$p(\mathbf{x}|\lambda)=\sum_{m=1}^Mw_mg(\mathbf{x}|\boldsymbol{\mu}_m,\boldsymbol{\Sigma}_m)$$
- 이 때 $g$ 는 각각의 Gaussian model이고, $\lambda$ 는 미지수들의 집합이다 ($M$ : Gaussian model의 수) : $$
\begin{gathered}
g\left(\mathbf{x} \mid \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m\right)=\frac{1}{(2 \pi)^{d / 2}\left|\boldsymbol{\Sigma}_m\right|^{1 / 2}} \exp \left[-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}_m\right)^T \boldsymbol{\Sigma}_m^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_m\right)\right] \\\\
\lambda=\left\{w_m, \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m\right\}, \quad m=1, \cdots, M
\end{gathered}
$$
#### Expectation-Maximization Algorithm (EM algorithm)

ML에서 미지수를 최적화하는 방법론 중 대표적인 gradient-descent 알고리즘과 더불어서 가장 많이 사용되는 EM algorithm이다. 

현재 우리가 추정해야 하는 미지수 family가 A와 B라는게 있다고 할 때, A와 B는 동시에 최적화할 수가 없는 상황이라고 하자. 그러면 이 작업을 하려면 A를 고정시키고, B만 최적화한다. 그런 다음 B를 고정하고 A를 최적화한다. A가 만약 바뀌었다면 / 갱신되었다면 그것을 고정하고 B를 최적화한다. 이러한 과정을 반복하면, A와 B가 불변/수렴하게 된다.

- MoG에서 이를 수행하는 방법 : 
	- A : $p(m|x)$ - 객체가 주어졌을 때, $m$ 번째 Gaussian 분포의 확률
	- B : $w_m,\mu_m,\Sigma_m$ 

- <font style="color:blue">E-step</font> : Given the current estimate of the parameters, compute the conditional probabilities
- <font style="color:blue">M-step</font> : Update the parameters to maximize the expected likelihood found in the E-step
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112113924.png)E-step에서는 Gaussian을 고정하고 각 Gaussian에 속할 객체의 확률을구하고, M-step에서는 객체의 확률을 고정하고, Gaussian parameter를 update하는 것

- Example : 
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/EM.gif)  
위 시뮬레이션을 보면 $M=2$개인 경우로, 처음에는 부적절한 가우시안 분포를 갖다가 시간이 지나면서 적절한 분포로 수렴하는 것을 볼 수 있다. 몇 개의 가우시안 분포, 즉 $M$을 몇으로 해야 가장 좋은지는 모르기 때문에 $M$ 을 바꾸어가며 EM algorithm을 수행하고 likelihood $L$을 최대화하는 $M$을 선택해야 한다.

#### EM algorithm for MoG 

- Expectation (E-step) : $$p\left(m \mid \mathbf{x}_i, \lambda\right)=\frac{w_m g\left(\mathbf{x}_i \mid \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m\right)}{\sum_{k=1}^M w_k g\left(\mathbf{x}_t \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)}$$
	- 분자 - $m$번째 분포에서 생성될 확률
	- 분모 - $M$개의 모든 분포에서 생성될 확률의 합
	- 여기서 $w_m, \mu_m$, $\Sigma_m$ 은 최적이 아닌 임의의 값으로 주어졌다고 가정하고 진행
- Maximization (M-step) : 
	- Mixture weight $$w_m^{(\text {new })}=\frac{1}{N} \sum_{i=1}^N p\left(m \mid \mathbf{x}_i, \lambda\right)$$
	- Means and Variances$$\boldsymbol{\mu}_m^{(n e w)}=\frac{\sum_{i=1}^N p\left(m \mid \mathbf{x}_i, \lambda\right) \mathbf{x}_i}{\sum_{i=1}^N p\left(m \mid \mathbf{x}_i, \lambda\right)}, \quad \sigma_m^{2(n e w)}=\frac{\sum_{i=1}^N p\left(m \mid \mathbf{x}_i, \lambda\right) \mathbf{x}_i^2}{\sum_{i=1}^N p\left(m \mid \mathbf{x}_i, \lambda\right)}-\boldsymbol{\mu}_m^{2(n e w)}$$
	- 이 때 $p(m|\mathbf{x}_i,\lambda)$는 고정
- The shape of MoG according to the covariance matrix : 
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112120809.png)
![](assets/(Mixture%20of)%20Gaussian%20Density%20Estimation/Pasted%20image%2020240112120818.png)

- MoG에서도 Full covariance가 가장 좋은 성능을 보일 것이지만, Full covariance $\Sigma$ 가 non-singular일 때만 가능하기 때문에 현실적으로는 어려움이 있다. 가장 좋은 대안은 Diagonal covariance로 진행하는 것이다.