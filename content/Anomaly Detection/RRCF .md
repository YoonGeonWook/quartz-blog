---
sticker: lucide//file-code
tags:
  - Anomaly_Detection
  - RRCF
---
# 🎲 Robust Random Cut Forest 

Robust Random Cut Forest (RRCF)는 이진 탐지 트리 (BST) 기반 알고리즘으로 input stream 데이터에서 이상 탐지를 목적으로 한다. 기존의 Isolation Forest (IF)를 실시간 스트리밍 환경에서 발생하는 이상 데이터를 탐지할 수 있도록 변형한 것이다. 

RRCF가 IF와 다른 것은 크게 두 가지이다 : 
1) Feature Selection : 변수 $p$ 를 선택할 때 uniform 분포에서 뽑는 대신, 변수들 각각이 갖는 range에 따라 뽑는 확률을 다르게 부여해 선택한다.
	- 이를 통해 실시간 스트리밍 환경에서도 이상 탐지가 잘 작동하도록 만들어줌
2) Anomaly Score : 평균 분기 횟수 (average path length) 대신에 Collusive displacement ($CoDisp$)라는 새로운 abnorm score를 사용한다.
	- 이상 데이터를 IF와는 다른 관점 (model complexity 관점)으로 정의하여 이상 탐지 성능을 향상


### Robust random cut tree (RRCT) 

RRCT는 RRCF를 구성하는 각 트리를 말한다. 주어진 sub-sampled data $S$ 에 대한 $\mathcal{T}(S)$는 아래와 같이 만들어진다.
- [ㅌ] RRCT 생성 알고리즘 : 
	1) Random select feature $p$ : 
		- 이때 $i$ 번째 feature가 선택될 확률은 $\frac{l_i}{\sum_j l_j}$ 이며, 여기서 $l_i = \max_{x\in S}x_i-\min_{x\in S}x_i$ .
		- 즉, 각 변수(feature) 값의 범위에 따라서 해당 변수가 선택될 확률이 달라짐
	2) Random select value $q$ : 
		- $[\min_{x\in S}, \max_{x\in S}]$ 의 범위를 갖는 uniform distribution에서 value $q$ 를 선택
	3) 객체 $x$의 선택된 변수값이 $q$ 이하이면 left child $S_1=\{x|x\in S,x_i\le q\}$ 로 분기, 그렇지 않으면 $S_2=\{x|x\in S, x_i>q\}$ 로 분기

이렇게 RRCT를 구축하고, 각 RRCT 들을 모은 것은 Forest가 RRCF가 된다. 

IF와 다른 점은 변수 $p$ 를 선택할 때 균등한 확률로 선택하는 것이 아닌 각 변수가 갖는 범위 (range)에 따라 다른 확률을 부여하여 선택한다는 것이다. 이 작은 차이만으로 시간에 따라 분포가 달라지는 실시간 스트리밍 환경에서의 데이터에 대응해 tree를 만들 수 있다. 

#### 실시간 스트리밍(real-time streaming) 환경

실시간 스트리밍 환경에서는 시간에 따라 유입되는 데이터의 분포가 달라진다. 과거 데이터를 학습한 모델이 이후 유입되는 데이터에도 좋은 성능을 보일 것이라는 보장은 없다. 따라서 새롭게 유입되는 데이터를 모델 학습에 사용해야 하는 것이다. 예를 들어, 현재 시점이 $t$ 일 때, 가장 최근 256개의 데이터로 tree를 만든다고 해보자. 이때 주어진 데이터는 $S_t=\{\mathbf{x}_{t-255},\mathbf{x}_{t-254},\cdots,\mathbf{x}_t\}$이고 이 $S_t$로 만든 RRCT는 $\mathcal{T}(S_t)$가 된다. 한 시점이 흘러 시점 $t+1$이 된다면 데이터 $S_{t+1} = \{\mathbf{x}_{t-254},\cdots,\mathbf{x}_t, \mathbf{x}_{t+1}\}$을 RRCT 생성 알고리즘으로 새로운 $\mathcal{T}(S_{t+1})$을 만들어야 한다. 하지만 이미 만들어놓은 $\mathcal{T}(S_t)$를 사용해서 이를 만들 수 있다면 효율적일 것이다. RRCT는 각 리프 노드가 (이론상) 데이터 포인트 한 개가 들어있는 tree이기 때문에 $\mathcal{T}(S_t)$에서 $\mathbf{x}_{t-255}$를 제거하고, $\mathbf{x}_{t+1}$을 추가해 $\mathcal{T}^\prime(S_{t+1})$을 만들 수 있다. 

- [ㅌ] 주의해야 할 점은 $\mathcal{T}(S_{t+1})$은 RRCT 생성 알고리즘으로 만든 RRCT이고, $\mathcal{T}^\prime(S_{t+1})$은 $\mathcal{T}(S_t)$를 변형해 만들 것이다.
	- [ ] RRCT 생성 알고리즘에 따라 변수 $p$와 value $q$의 임의 선택에 따라 무수히 많은 $\mathcal{T}(S_{t+1})$을 만들 수 있다. 
		- 즉, $\mathcal{T}(S_{t+1})$는 확률 변수 (random variable)이며 확률 분포를 갖는다
	- [ ] 반면, $\mathcal{T}^\prime(S_{t+1})$은 $\mathcal{T}(S_t)$에서 데이터를 추가/삭제함으로써 변형한 것이기 때문에 유일하게 결정된다. 
		- [ ] 즉, $\mathcal{T}^\prime(S_{t+1})$은 $\mathcal{T}(S_t)$에 의존적/종속적이다. 
		- [ ] $\mathcal{T}(S_t)$는 확률 변수이기 때문에, $\mathcal{T}^\prime(S_{t+1})$는 그로 인해 확률 분포를 갖는다 - 확률 변수의 변형이므로

이러한 이유로 $\mathcal{T}(S_{t+1})$의 확률 분포와 $\mathcal{T}^\prime(S_{t+1})$의 확률 분포가 같을 것이라 기대하기는 힘들다. 그 이유는 $\mathcal{T}(S_{t+1})$는 $S_{t+1}$의 각 변수들의 범위로 만들어지고, $\mathcal{T}^\prime(S_{t+1})$는 $S_t$의 각 변수들의 범위로 만든 tree에 노드의 삭제/추가라는 변형만 추가된 것이기 때문이다. 실시간 스트리밍 환경에서 새로 유입되는 데이터로 학습한 모델 $\mathcal{T}(S_{t+1})$과는 다른 분포를 갖는 모델 $\mathcal{T}^\prime(S_{t+1})$을 사용한다는 것은 엉뚱한 tree들로 구성된 RRCT로 데이터의 anomaly score를 계산한다는 것이다. 

하지만 이 논문에서 추가로 제안하는 알고리즘을 사용하면 $\mathcal{T}(S_{t+1})$와 $\mathcal{T}^\prime(S_{t+1})$의 확률 분포가 같아진다고 한다. 

### Displacement (Disp)

IF에서는 이상 데이터에 대해서는 iTree의 분기 횟수가 적을 것이라는 것에 주목해 anomaly score를 계산했다. RRCF는 이와 다르게 이상 데이터가 존재가 모델의 복잡성을 증가시킨다는 것에 주목했다. 

정상 데이터만 있을 경우 모델은 이들의 분포만 학습하면 된다. 이상 데이터는 정상 데이터와 전혀 다른 분포를 갖는다는 전체 하에, 이상 데이터의 존재는 학습 모델로 하여금 정상 데이터의 분포 뿐 아니라 이상 데이터의 분포도 학습해게끔 만든다. 따라서 정상 데이터만을 학습할 때와 비교해 모델의 복잡성이 증가하게 되는 것이다. 
![](assets/RRCF/Pasted%20image%2020240123194743.png)
위 그림은 원형(파란색) 데이터로 만든 RRCT와, 노란색 점을 추가해 만든 RRCT를 비교해 늘어난 범위 (빨간색)를 비교해 증가된 분기 수를 보여준다. 노란색 점이 추가될 경우 이 포인트를 isolation시키기 위해서 늘어난 빨간색 범위 안에서 적어도 한 번 이상의 split이 더 필요해 보인다. 운 좋게 추가 첫 split에서 노란색 점을 isolation 할 수 있다면, 원래의 파란색 데이터로 만든 RRCT의 최상단에 split 하나를 더 추가한 tree가 만들어질 것이다. 따라서 이상 데이터 (노란색)가 추가된 RRCT는 모든 파란색 데이터들의 path length를 1 증가시킨다. 

추가 첫 split이 아닌 $d$ 번째 split에서 노란색 점이 isolation 된다면 $d-1$ 번째까지 노란색 점과 함께 split되어 왔던 데이터들은 $d$ 번째 split에서 분리되면서, 기존 파란색만의 RRCT와 비교해 path length가 1 증가하게 된다. 

위 그림처럼 정상 데이터와 멀리 떨어진 노란색 점과 같은 데이터 포인트가 추가될 경우에 기존 RRCT를 구성하는 정상 데이터들의 path length는 증가한다. 반대로 노란색 점과 같은 데이터 포인트가 정상 데이터 부근에 존재한다면 path length가 증가하는 데이터는 그리 많지 않을 것이다. 

- [ㅌ] RRCF에서 사용하는 anomaly score는 Disp (displacement)라는 것이다. 주어진 sub-sampled data $S$의 한 데이터 $x$ 의 anomaly score $Disp(x,S)$는 아래와 같이 정의된다.
	- [ ] 데이터 $S$ 로 만든 RRCT에서 데이터 $x$ 를 제거했을 때, 나머지 데이터에서 발생하는 depth 변화의 총합
	- [ ] 데이터 $S$ 로 만들 수 있는 RRCT는 매우 다양할 수 있기 때문에 각 RRCT 에서 $x$ 를 제거했을 때 발생하는 depth 변화의 총합에 기대값을 취한 것이 $Disp(x,S)$가 된다.

아래 그림을 보면서 하나의 RRCT에서 데이터 $x$에 대한 $Disp(x,S)$ 값을 살펴보자. 하나의 RRCT에서 $x$ 를 제거하면 발생하는 depth 변화의 총합은 **데이터 $x$의 자매노드에 있는 데이터의 개수**이다. 따라서 $Disp(x,S)$는 데이터 $S$로 만든 여러가지 RRCT에서 $x$의 자매 노드에 있는 데이터 개수의 평균값이다.

![](assets/RRCF/Pasted%20image%2020240124143227.png)RRCT에서 $x$를 제거하면 subtree $c$ 안에 있는 노드들의 depth가 1씩 감소한다. 한편, subtree $b$의 depth는 변하지 않는다. 따라서 $x$의 자매노드에 있는 데이터 개수가 depth 변화의 총합이 된다. $x$가 anomaly일수록 $x$로 인한 전체 depth 변화는 클 것이다.

### Collusive Displacement (CoDisp)

위에서 설명한 $Disp$를 anomaly score로 바로 사용하면 masking 문제가 발생한다. Masking은 anomaly끼리 모여서 마치 정상인 것처럼 정체를 감추는 현상을 말한다. 이상 데이터 $x$ 바로 옆에 $x^\prime$이 하나 더 있다면, $x$의 자매 노드에 $x^\prime$ 하나만 존재할 것이다. 따라서 $Disp$ 값은 1 정도일 것이다. 따라서 이 논문에서는 $x$의 정체를 숨기려고 하는 공모자 (colluder)들까지 고려하는 anomaly score인 <font style="color:skyblue">Collusive Displacement (CoDisp)</font>를 제안했다. 


이상 데이터 $x$의 공모자(colluder) 집합을 $C$라고 하자. 데이터셋 $S$로 만든 RRCT에서 $x$만 제거하는 것이 아니라, 집합 $C$를 제거했을 때 발생하는 depth 변화량의 총합을 anomaly score로 고려할 것이다. 즉, $x$ 에 대한 anomaly score로 $Disp(x,S)$가 아닌 $Disp(C,S)/|C|$를 사용하는 것이다. 이 때 집합 $C$의 크기가 클수록 (공모자가 많을수록) RRCT에서 $C$를 제거할 때 tree의 변화가 커질 것이다. 이는 anomaly score 값이 공모자의 수에 영향을 받지 않도록 집합 $C$의 크기로 나눠준 것이다.

하지만 우리는 공모자 집합 $C$를 미리 알 수 없다. 논문에서는 $x$를 포함하는 가능한 모든 부분집합을 고려한다. 그리고 $Disp(C,S)/|C|$의 최대값을 anomaly score로 사용한다. 
$$
CoDisp(x,S) = \mathbb{E}_T\left[\max_{x\in C\subset S}\frac{Disp(C,S)}{|C|}\right]
$$
기대값은 tree $T$로써 데이터셋 $S$로 만든 RRCT이다. 여러가지 RRCT에 대해 계산한 후 평균 내서 $CoDisp(x,S)$를 계산하는 것이다. 물론 가능한 모든 subset $C$를 고려한다는 것은 불가능하다. 따라서 실제 구현에서는 RRCT 안에서 $x$ 부모겪인 데이터들만 $C$로 간주하여 연산을 진행한다. 이렇게하면 공모자 $C$를 제거했을 때의 모델의 depth 변화의 총합을 $C$의 자매노드에 있는 데이터의 개수로 구할 수 있다. 

- [ㅌ] 하나의 RRCT에서 데이터 $x$의 $CoDisp$ 값은 아래와 같이 구한다. 
	- [ ] $x$의 자매 노드에 있는 데이터 개수 / 1
	- [ ] $x$의 부모 노드의 자매 노드에 있는 데이터 개수 / 부모 노드의 크기
	- [ ] $x$의 조부모 노드의 자매 노드에 있는 데이터 개수 / 조부모 노드의 크기
	- [ ] $\vdots$

이들 중에서 최대값을 구하면 된다. 그런 다음 여러 개의 RRCT로 부터 이 최대값들을 모두 구한 다음 평균을 내면 그 값이 $x$에 대한 $CoDisp$가 된다. 

- 이 $CoDisp$ 값이 클수록 anomaly로 간주한다.



### 최종 알고리즘

- [ㅌ] Input : 
	- [ ] $Z$ : 주어진 데이터셋
	- [ ] `num_trees` : RRCT 개수
	- [ ] `tree_size` : size of sub-sampling

1. Construct Forest - Training
	1)  RRCT의 모음 RRCF 저장 객체 생성 : `forest = []`
	2) `num_trees`개의 RRCT를 생성하는데, 각 RRCT는 $Z$에서 `tree_size` 크기로 만든다
2. 새로운 데이터 $x$의 $CoDisp$ 구하기
	1) 여러 개의 RRCT의 $CoDisp$ 값을 저장할 객체 생성 : `codisp = []`
		2) `forest`에 있는 각 RRCT에 $x$를 흘려보낸 다음
			3) 각 RRCT에서 $x$의 조상(본인, 부모, 조부모, ...)들에 대해 $Disp(C, Z)/|C|$ 계산
		4) 모든 조상에 대한 $Disp(C,Z)/|C|$ 중 최대값을 `codisp`에 추가
		5) RRCT에서 $x$ 제거
	6) `mean(codisp)`를 $x$에 대한 최종 $CoDisp$로 산출


### 실험

이 논문에서는 인위 데이터데 대한 실험 두 가지와 실제 데이터에 대한 실험 한가지가 있다. 인위 데이터에 대한 실험은 IF가 이상 데이터를 적절히 잡아내지 못하는 상황을 제시하며 진행된다.

#### 인위 데이터 1. 의미 없는 축이 굉장히 많은 경우

- [ㅌ] Training set은 차원이 30이고, 행이 2,010개로 이루어져 있다 : 
	- [ ] 1,000개의 데이터는 첫 변수값만 +5이고, 나머지 29개는 0인 벡터에 gaussian noise가 추가된 벡터다: $$\begin{aligned}\mathbf{x}_i=(5,0,0,\cdots,0)^T+\epsilon_i\\ where\quad \epsilon_i\sim\mathcal{N}(\mathbf{0}_{30},\mathbf{I}_{30})\\ for \quad i=1,\cdots, 1000\end{aligned}$$
	- [ ] 또 다른 1,000개의 데이터는 첫 번째 원소가 -5이고, 나머지 29개는 0인 벡터에 gaussian noise가 추가된 벡터다 : $$\begin{aligned}\mathbf{x}_i=(-5,0,0,\cdots,0)^T+\epsilon_i\\ where\quad \epsilon_i\sim\mathcal{N}(\mathbf{0}_{30},\mathbf{I}_{30})\\ for \quad i=1,\cdots, 1000\end{aligned}$$
	- [ ] 나머지 10개의 데이터는 gaussian noise 벡터로 이상 데이터를 나타낸다 : $$\begin{aligned}\mathbf{x}_i=\epsilon_i\\ where\quad \epsilon_i\sim\mathcal{N}(\mathbf{0}_{30},\mathbf{I}_{30})\\ for \quad i=1,\cdots, 1000\end{aligned}$$
이러한 데이터에서 첫 번째 변수를 제외한 29개의 변수값이 노이즈로 구성되어 있기 때문에 정상 데이터와 이상 데이터를 구분하기 어렵다. 따라서 tree model이 이상 데이터를 isolation시키기 위해서는 반드시 첫 번째 축을 선택해야 한다. 이때 IF는 모든 축을 동일한 확률로 선택한다. 따라서 $\frac{29}{30}$의 확률로 의미 없는 축/변수를 선택해 tree를 만들 것이다. 이로 인해 정상이든 이상이든 상관없이 path length가 증가하면서 anomaly score는 감소할 것이다. 즉 IF는 의미 없는 축이 많은 경우에 이상 데이터를 잘 잡아내지 못한다. 

한편, RRCF는 각 축이 갖는 범위에 따라 축을 선택할 확률이 달라지므로 높은 확률로 첫 번째 축을 계속 뽑으며 이상 데이터를 고립시킬 수 있다. 

이제 이상 데이터 10개 없이 2,000개의 데이터만을 사용해 Forest를 만든다고 해보자. 그 후 영벡터 $\mathbf{0}\in \mathbb{R}^{30}$가 test data로 들어온다고 해보자. IF는 새로 들어온 데이터가 각 트리에서 어디에 위치할지 root node부터 split을 따라가며 추적한다. 어느 순간 이후 부터는 split이 +5 주변에서만 또는 -5 주변에서만 일어나게 될 것이다. IF는 training data의 값의 범위만을 사용해 tree를 만들기 때문이다. iTree 훈련 당시 영벡터는 없었기 때문에 +5/-5 두 군집이 나위고 나서부터는 한 노드에서 +5 주변, 다른 노드에서는 -5 주변에서만 split 기준을 선택해 iTree를 구성하게 된다. 결과적으로 영벡터는 미리 만들어 둔 iTree에서 고립이 잘되지 않기 때문에 anomaly score가 굉장히 작을 것이다. 

반면 RRCF의 경우 새로 들어온 데이터를 미리 만들었던 RRCT에 추가하는 알고리즘이 있다. 위에서 노드의 삽입/제거 알고리즘을 다루진 않았지만, 이상 데이터를 RRCT에 추가할 때 1) 해당 데이터를 고려한 값의 범위에서 split 기준을 다시 설정해보고, 2) 기존 데이터로부터 고립이 되는지 확인 과정이 있다. 3) 이 과정을 통해 유입된 데이터가 고립되는 split을 찾아 새롭게 노드로 추가된다. 

따라서 training 과정에서 보지 않은 데이터에 대해서도 tree를 새롭게 만들어 anomaly score를 계산할 수 있게 된다. 위 실험은 아래 그림을 통해 IF와 RRCF의 anomaly score를 비교했다. 

![](assets/RRCF/Pasted%20image%2020240124155233.png)IF (왼쪽)의 경우 거의 모든 데이터에 대해 anomaly score가 0.3을 넘지 못하는데, 이는 의미 없는 축으로 split 횟수가 증가했기 때문이다. 그로 인해 이상 데이터와 정상 데이터가 잘 구분되지 않는다. RRCF (오른쪽)는 이를 굉장히 잘 구분하고 있다. 

#### 인위 데이터 2. 실시간 스트리밍 데이터

두 번째 실험은 아래 코드를 사용해 인위 데이터를 생성해 진행되었다. 이 코드는 파이썬 [`rrcf`](https://klabum.github.io/rrcf/)에서 제공된다. 730일 동안 sin 함수를 따르는 신호가 기록되어 있고, 235일부터 255일까지는 이상 신호가 발생한 것으로 이해하면 된다.

```python 
n = 730 
A = 50
center = 100
phi = 30
T = 2 * np.pi / 100
t = np.arange(n)
sin = A * np.sin(T * t - phi * T) + center
sin[235:255] = 80
```

- [ㅌ] 두 가지 중요 포인트 : 
	- [ ] 730일 데이터가 한 번에 주어지는 것이 아니라, 하루에 하나씩 데이터가 들어오는 <font style="color:skyblue">실시간 스트리밍 환경</font>을 가정
	- [ ] `Shingling`이라는 방법으로 1차원 데이터를 4차원 데이터로 바꿔준 후 RRCT를 만듦
		- `Shingling` : 최근 $k$개의 값을 열벡터로 결합해 feature 벡터로 사용하는 방법이다. 예를 들어 크기가 4인 `shingling`을 사용할 경우, 첫 번째 데이터는 $(t_1,t_2,t_3,t_4)^T$, 두 번째 데이터는 $(t_2,t_3,t_4,t_5)^T, \cdots$ 이런 식으로 데이터가 구성됨
		- 시계열 데이터 분석에서 자주 사용되는 방법

실시간 스트리밍 환경에서는 데이터가 하나씩 들어올 때마다 `num_tree`개의 모든 RRCT에 데이터를 추가해준다. 그러다가 tree의 크기가 `tree_size`에 도달하면 가장 과거의 데이터를 제거해주고 새로운 데이터를 추가하는 방식으로 Forest를 유지한다. 논문에선 `num_tree = 100`, `tree_size = 256`을 사용했다. 

![](assets/RRCF/Pasted%20image%2020240124160103.png)

위 그림은 이 실험의 결과인데, 파란색 선이 sin 그래프에서 235~255 구간을 이상 신호로 변환한 것이다. 빨간색 선은 IF와 RRCF의 anomaly score를 나타낸다. 이 값이 높으면 이상 데이터로 간주하는 것이다. IF의 경우 이상 신호가 모두 끝난 후에야 anomaly score 값이 높아진다. 반면에 RRCF의 경우 이상 신호의 시작과 종료 모두 anomaly score가 높다. 즉, RRCF는 실시간 스트리밍 데이터에 적합하며, 이상 신호의 시작을 포착하는데 탁월하다는 것이다.

#### 실제 데이터 1. 뉴욕시 택시 탑승객 수 데이터

마지막 실험은 "뉴욕시 택시 탑승객 수" 데이터셋에 RRCF를 적용한 것이다. 해당 데이터셋은 unsupervised learning 이상 탐지 분야에 자주 사용되는 벤치마킹 데이터셋 [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) 데이터셋 중 하나이다. 데이터셋에는 2014년 7월부터 2015년 1월까지의 뉴욕시 택시 탑승객 수가 30분 단위로 저장되어 있다. 원래는 레이블이 없는 데이터지만 논문에서는 연휴나 기념일 등 총 8개의 이벤트를 이상신호로 간주해 정량적인 평가도 더했다. 이때, 특정 일에 포함되는 데이터 모두 (하루에 48개)를 이상 데이터로 레이블링했다 : 
- **Independence Day** (2014-07-04 ~ 2014-07-06)
- **Labor Day** (2014-09-01)
- **Labor Day Parade** (2014-09-06)
- **NYC Marathon** (2014-11-02)
- **Thanksgiving** (2014-11-27)
- **Christmas** (2014-12-25)
- **New Years Day** (2015-01-01)
- **North American Blizard** (2015-01-26 ~ 2015-01-27)

이 데이터셋도 시계열 데이터이므로 size 48의 shingling을 사용했다. 즉, 과거 48개 (총 24시간; 하루)의 탑승객 수를 결합해 하나의 데이터를 만든 것이다. 그리고 이 실험 역시 실시간 스트리밍 환경을 가정하고 진행했다. 논문에서는 `num_tree = 200`, `tree_size = 1000`를 사용했다. 아래 그림은 실험 결과를 나타낸다. 파란색 선은 탑승객 수를, 빨간색 선은 anomaly score를 나타낸다. 몇 가지 주요 이벤트에 대해 높은 anomaly score를 보이고 있다. 한 가지 유의할 점은 2014년 7월 14일부터 2014년 9월 15일까지의 결과가 없다는 것이다.  2014년 9월 16일부터는 총 5개의 abnormal event가 있었는데, **Thanksgiving**을 제외하고 나머지 4개 이벤트를 성공적으로 탐지해냈다. 

![](assets/RRCF/Pasted%20image%2020240124163958.png)

아래 표는 IF와 RRCF 모델의 정량적인 평가를 나타낸다. IF 모델에 대해서는 논문 저자가 직접 실시간 스트리밍 환경을 만들어서 사용했는데, 그 방법이 쫌 나이브하다보니 IF의 성능 지표가 낮게 나온 것일 수 있다. 여러 지표들 중 RRCF는 precision에서 큰 차이를 만들어 냈다. 이는 모델이 anomaly라고 예측한 것 중 맞춘 비율이 높음을 의미한다. 다르게 표현하면 모델이 잘못된 경보를 울리는 비율이 작다는 것이다. 

![](assets/RRCF/Pasted%20image%2020240124164339.png)뉴욕시 택시 탑승객 수 데이터셋에 대한 IF와 RRCF 정량적 평가 (데이터 단위의 평가)

아래 표는 이벤트 단위로 점수를 매긴 것이다. 하나의 이벤트는 하루에서 길게는 3일로 구성된다. 그리고 하루마다 30분 단위로 48개의 데이터가 기록된다. 위의 표는 이 30분 단위마다 정상/이상을 예측해 평가지표를 계산한 것이고, 아래 표는 이벤트 단위로 이상 감지를 성공했는지를 나타내는 표이다.

주목할 점은 Time to detect onset/end인데, 이는 각각 이벤트를 이상 데이터라고 감지하기 시작한 시점과 종료한 시점을 말한다. 30분 단위인 것을 고려하면 IF는 이벤트 발생 후 평균적으로 약 11시간만에 이상 데이터라고 예측했고, RRCF는 이벤트 발생 후 평균적으로 약 7시간만에 이상 데이터라고 예측을 했다. 이상 감지까지 너무 오래 걸렸다고 생각들 수 있지만, 휴일/행사가 시작되자마자 택시 탑승객 수가 급격히 바뀌는 것이 아니기 때문에 이는 꽤 빠르게 감지한 것으로 볼 수 있다. 
![](assets/RRCF/Pasted%20image%2020240124164716.png) 뉴욕시 택시 탑승객 수 데이터셋에 대한 IF와 RRCF 정량적 평가 (이벤트 단위 평가)

