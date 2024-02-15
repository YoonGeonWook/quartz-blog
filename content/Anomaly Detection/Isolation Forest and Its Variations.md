---
sticker: emoji//1f4d5
tags:
  - Anomaly_Detection
  - 이상탐지
  - Isolation-Forest
banner: Anomaly Detection/image/IForest.png
---
# 🎲 Isolation Forest and Its Variations

## 🎯 Isolation Forest
&#9495; Liu et al. (2008, 2012) 
---

#### Anomaly detection의 특성 및 문제점

- Anomaly detection은 문제의 특성상 class의 imbalace가 굉장히 심함
- 또한 anomaly/novelty마다 다른 특성을 가질 수 있기 때문에 normal이 아닌 객체를 하나의 패턴으로 묶어서 분류하는 것은 잘못된 방식일 수 있음
- 많은 양의 데이터에 대해 normal/abnormal이라고 labeling하는 것은 힘들기 때문에 일반적인 분류 문제와 같은 supervised learning 보다는 unsupervised learning으로 접근하는 방법을 사용함

Unsupervised learning을 통한 anomaly detection의 많은 기존 방식을은 학습 데이터의 normal을 이용해 객체간 density/distance를 기반으로 한 계산을 통해 정상 영역을 정의한 후 Test 데이터를 이에 적합하여 정상 영역에서 벗어난 정도를 abnormal scores로 측정한다. IForest 저자는 이와 같은 방식에는 2가지 단점이 있다고 한다. 

1) 기존 모형들은 normal 데이터에 최적화되어 있어서 anomaly detection 성능이 떨어짐
2) 모든 객체들에 대해 density를 계산하는 것은 computational cost가 매우 높아서 대용량 데이터셋이나 고차원 데이터에는 적용이 어려움

저자는 위와 같은 문제를 Isolation Forest를 통해 극복할 수 있다고 주장한다. 모델링에 들어가는 시간 (time complexity)이 선형적이라고 주장한다. 그래서 대용량 혹은 고차원 데이터에도 충분히 적용이 가능하다고 말한다.  


- [ㅌ] Assumption : <font style="color:skyblue">Few and Different</font>
	1) 소수 범주 (minority), 다시 말해서 anomaly/abnormal/novel data는 훨씬 더 적은 개체로 존재할 것이다. 
		- 즉 전체 데이터에서 anomaly가 차지하는 비율이 작다.
	2) 그 minority 객체들은 normal data/instances 와 매우 다른 속성값 (attribute-values)을 가질 것이다.

어떤 Tree를 만든 후 이를 이용해서 특정 객체 하나를 고립시킬 수 있다면 (Isolation 이라고 부르는 이유임) novel data에 대해서 쉽게 고립시킬 수 있을 것이다. 여기서 쉽게 고립시킨다는 것은 Tree가 분할 (split)을 몇 번 하지 않아도, 즉 적은 수의 split으로 isolation을 할 수 있다는 것이다.


- [ㅌ] 모든 단일 인스턴스를 효율적으로 고립시킬 수 있는 구조를 갖는 Tree 
	- [ ] Novel instances는 Tree의 root에 가까이 고립됨
	- [ ] Normal instances는 Tree의 고립이 상대적으로 어려워서, 많은 split이 필요함 - isolated at the deeper end of tree

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120145646.png)

위 그림에서는, 파란색으로 표시된 $x_i$ 와 빨간색 $x_o$ 객체를 고립시키는 tree를 만든다. 이 tree를 만드는 과정은 상당히 간단하다. 기존의 의사결정나무에서는 (분류or회귀) 정보획득 (IG)이 크게 되는 변수와 기준점을 찾아서 분할을 한다. 하지만 IForest에서는 임의의 변수를 선택한 후 그 변수의 임의의 값을 사용해서 split을 한다. 한 번 split을 한 다음 isolation 하고 싶은 객체가 어느 쪽에 있는지 확인하고 그 객체가 속하지 않은 부분을 버리고 진행한다. 

오른쪽 빨간색 $x_o$ 를 먼저 살펴보자. ①번 째 기준선을 임의로 정한 후 $x_o$ 가 그 아래 있으므로 이 선의 윗쪽은 버린다. 그 다음 ②번 기준선을 임의로 정한 뒤 $x_o$ 가 오른쪽에 있으므로 기준선의 왼쪽은 버린다. 이 과정을 반복한다. 마지막 기준선을 선택한 후 $x_o$ 가 속한 영역이 isolation 되어 있음을 알 수 있다. 그럴 때 마지막 기준선 ④의 값을 기록한다. 

왼쪽 파란색 $x_i$ 도 마찬가지 과정을 반복하는데, 이 객체는 밀집된 영역의 가운데 있으므로 동일한 과정을 굉장히 여러 번 반복하게 된다. 

IForest 에서는 특정 객체를 고립시키는 데 몇 번의 split이 필요한지를 측정해서 이 횟수를 anomaly score로 사용한다. Split 횟수가 적으면 anomaly score가 크고, 적으면 score가 작다. 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120144748.png)

위 그림을 보면 x 축이 log scale인데, 최종적으로 1,000번 (num of trees) 반복한다. 100개 이상의 trees를 사용할 경우 $x_i$ 와 $x_o$ 를 고립시키는데 필요한 split의 횟수가 뚜렷한/유의미한 차이를 보인다. 

##### Isolating an abnormal instance
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120150702.png)

##### Isolating a normal instance
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120150759.png)
##### The isolation characteristics of tree forms the basis of the method to detect novel instances

IForest 는 충분히 많은 isolation trees를 만들어서 이들이 갖는 scores를 집계 (aggregate)하면 일반화된 성능과 높은 판별력을 가질 수 있다. 
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120151121.png)

#### Definition : Isolation Tree - iTree
- [ㅌ] 어떤 샘플 데이터 $X$ 가 주어졌을 때, <font style="color:skyblue">임의의 속성/변수 $q$ 와 그것의 split value $p$ 를 재귀적으로 분할</font>을 하는데, 아래 조건을 만족할 때까지 진행한다 : 
	- [ ] Tree가 height limit/max depth 에 도달
	- [ ] $|X|=1$ ⇒ isolation 
	- [ ] 주어진 영역 안에 두 개 이상의 객체가 완벽하게 같은 값을 가질 경우 ⇒ 아무리 분할해도 고립시킬 수 없음

위 3가지 조건 중 두 번째가 isolation에 해당하는 개념이고, 첫 번째와 세 번째는 알고리즘의 효율성을 추구하는 조건이다. 
#### Definition : Path Length
- [ㅌ] Path length $h(x)$는 객체 $x$ 가 root node에서 terminal node까지 가는데 걸리는 edge의 수 (즉 몇 번 split 했는지)를 측정하여 계산한다.
- [ㅌ] 이 split 횟수 $h(x)$ 의 평균 (average)은 아래와 같은 Euler's constant로 normalized 된다: $$c(n) = 2H(n-1)-\frac{2(n-1)}{n},\quad H(a) = \ln(a) + 0.5772156649$$
	- [ ] 이 Euler's constant로 normalized 된 average path length of $h(x)$ 는 novelty score에 사용된다.

#### Definition : Novelty score
- [ㅌ] 객체 $x$ 의 novelty score는 아래와 같이 정의된다 : $$s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}$$
	- 여기서 $c(n)$ 은 평균 path length을 나타내고, $h(x)$는 1개의 iTree에 대해 $x$ 를 isloation 시키기 위한 path length
	- [ ] $E(h(x))\to c(n) \Longrightarrow s\to 0.5$
	- [ ] <font style="color:orange">$E(h(x))\to0 \Longrightarrow s\to1$</font>
		- isolation이 매우 쉬울 경우
	- [ ] <font style="color:skyblue">$E(h(x))\to n-1 \Longrightarrow s\to 0$</font>
		- isolation이 매우 어려울 경우

다시 말해서 path length가 짧을수록 anomaly score $s$ 는 1에 가깝고, 길수록 $s$ 는 0에 가깝다.

#### Novelty score contour (example) 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120153121.png)

오른쪽 그림은 거의 대부분의 정상 데이터는 원형의 띠에 있고, 한 가운데 있는 group outlier를 가진다. 

#### Isolation Forest - pseudo code

- [ㅌ] Randomly sample datasets
- [ㅌ] Construct iTree
- [ㅌ] Compute the path length

아래에 나오는 Algorithm 1 & 2는 IF 알고리즘의 Training stage 이고, Algorithm 3은 Evaluation stage이다. 

- Training stage에선 input $X$를 이용해 $t$개의 sub-samples와 iTree를 만들어 iForest로 ensemble 해준다.
- Evaluation stage에선 Training stage에서 생성한 $t$개의 iTree에 대해 모든 데이터 포인트 $x$의 path length를 계산한다. 이 path length를 기반으로 각 데이터 포인트의 anomaly scores를 산출한다.

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122212220.png)
1. iForest를 저장할 빈 객체 생성
2. Tree의 height/depth limit $l$ 설정
3. For Loop : sub-sampling $t$ 회 반복 - boostrapping without replacement
	4. sub-sampling with size $\psi$
	5. iTree construction & iForest에 저장
6. end For Loop
7. iForest 반환
#### Training IForest
- [ㅌ] <font style="color:skyblue">Randomly sample datasets</font> : <font style="color:skyblue">256 is generally enough</font>
	- [ ] 주어진 데이터셋이 클 경우, IForest의 계산 복잡도가 매우 높아진다
	- [ ] 특정 객체를 isolation 시킬 때, 그 객체가 아닌 reference/training data의 모든 객체를 쓰지 않아도 된다
	- [ ] 대략 256개 정도만 sampling 해서 사용하면, 계산 복잡도는 충분히 줄어들지만 성능에는 큰 차이가 없다

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120154110.png)

Target 데이터 (고립시키고자 하는 객체)는 고정시켜두고, 원래의 데이터셋에서 256개 정도의 일부분만 sampling해서 isolation을 진행한다. 두 번째 itree에서는 target을 제외한 나머지 256개는 다시 sampling 한 뒤 isolation을 진행한다. 이러한 과정을 num of trees 만큼 반복하기 때문에 256개를 임의로 sampling 하는 것은 결국 전체 분포를 반영하게 되는 효과를 얻는다. 

- [ㅌ] <font style="color:skyblue">Construct iTree</font>
	- [ ] 핵심은 임의의 변수/속성 $q$ 와 임의의 split point $p$ 를 선택하는 것
	- [ ] 최종적으로 isolation 하고자 하는 객체가 들어있는 부분만 보존한다

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120154539.png)
1. sampling된 $X^\prime$이 isolation 되었다면 
	2. external/terminal node로 반환
3. 그렇지 않다면 
	4. $X^\prime$을 리스트 $Q$의 원소로 저장
	5. 리스트 $Q$의 원소 중 하나 $q$ 를 임의로 선택
	6. split에 사용될 criterion 값 $p$ 를 임의로 선택 - $X^\prime$가 갖는 변수 $q$의 range 안에서
	7. $q<p$인 $x$들은 $X_l$로 할당
	8. $q\ge p$인 $x$들은 $X_r$로 할당
	9. 모든 데이터 포인트가 isolation 될 때까지 이 과정을 반복
		- split 될 때마다 split 정보 $(\text{attribute } q, \text{ split value } p)$  저장

- [ㅌ] <font style="color:skyblue">Compute the path length</font>
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120154824.png)
Training stage에서 만들어진 iTree $T$에 데이터 포인트 $x$를 input한다.
1. 현재 $x$가 있는 노드가 terminal node이거나 $e > hlim$이면 
	2. $e+c(T.size)$를 반환
		- $e$ : 객체 $x$가 현재 있는 노드의 path length
		- $c(T.size)$ : 현재 노드에 있는 데이터 포인트 개수로, 계속 split 했을 때 기대되는 기대되는 path length - $c(n)$ 
			- 이 값을 더해주는 이유는 Training stage에서 $hlim$에 걸려 tree split이 멈춘 경우 그 노드를 external/terminal node로 반환하는데, 그 노드 안에 1개가 아닌 여러 개의 데이터 포인트가 존재할 수 있다. split을 멈추지 않고 계속했을 때의 사전에 기대되는 path length를 $c(T.size)$로 계산해 더해준다.
4. 현재 iTree $T$에 저장되어 있는 split attribute을 $a$로 선언
5. 데이터 포인트 $x$의 변수 $a$값을 iTree $T$에 저장되어 있는 split value $p$ 와 비교:  만약 $x_a < T.splitValue$ 이면 
	6. 현재 노드 $T$ 의 왼쪽 노드인 $T.left$로 보내고, 다시 $PathLength(\cdot,\cdot,\cdot)$ 알고리즘 수행
7. 그렇지 않고 $x\le T.splitValue$ 이면
	8. 현재 노드 $T$의 오른쪽 노드인 $T.right$로 보내고, 다시 $PathLength(\cdot,\cdot,\cdot)$을 수행

- [ ] 이 Algorithm 3를 통해 모든 데이터 포인트의 path length를 구할 수 있다. 이를 $t$ 개의 iTree들에 적용한 후 평균내면(average) 모든 데이터 포인트에 대해 average path length $E(h(x))$를 얻을 수 있다. 이를 이용하여 anomaly score $s(x,n)$을 계산한다.

이렇게 구한 $s(x,n)$은 $[0,1]$ 의 값을 보인다. 이 값을 기준으로 데이터 포인트를 정렬하면 랭킹을 매길 수 잇다. 이 랭킹을 기준으로 abnormal이 의심되는 데이터 포인트를 추려낼 수 있다. Anomaly로 판별하는 특정 threshold 값을 지정할 수도 있는데, 이는 데이터의 상태와 분석가의 판단으로 결정된다. 혹은 threshold를 설정하지 않고 랭킹이 높은 데이터 포인트들을 개별적으로 분석해서 anomaly인지 판단할 수도 있다. 이렇게 anomaly scores를 계산하는 것은 전체 데이터 중 anomaly가 의심되는 데이터 포인트를 구별하는 데 드는 모니터링 비용을 감소할 수 있다. 

#### Effect of the height limit

중요한 hyperparameter 중 하나는 iTree 를 종료시키는 조건 중 하나인 height limit / max depth이다. 아래 그림을 보면 $hlim$ 을 6과 1로 둔 경우를 비교했는데, 어느 정도 split을 시켜야 등고선 (contour)이 그럴듯하게 나온다. 너무 짧은 height limit을 설정하면 원하지 않는 경우가 발생할 수 있다. 보통 depth는 두 자리 수 정도로 잡으면 문제없이 작동한다. 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120155102.png)

#### Empirical Evaluation
- [ㅌ] Datasets : 
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120155221.png)

- [ㅌ] Performance - AUROC : 
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120155301.png)

- [ㅌ] Performance - computational complexity : 
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120155427.png)


## 🎯Extended Isolation Forests
&#9495; Hariri et al. (2018)
---

#### 기존 Isolation Forest

IF는 feature space를 재귀적인 임의 분할을 하는 iTree를 여러 개의 subsamples로 만든 후, 구축된 iTree 전부를 사용해 iForest를 통해 각 객체별로 anomaly scores를 계산한다. 이러한 IF는 computational cost가 효율적이고 이상탐지의 성능이 좋다는 장점이 있다. 

- [ㅌ] Extended Isolation Forest : 
	- [ ] EIF는 기존 IF 구조를 바꾸지 않고 그대로 사용한다. 다만 iTree 건설에 사용하는 random split의 형태를 약간 변형했다. 기존의 random split은 모든 변수 중 특정한 하나를 임의로 선택해 그 변수의 범위(ranger) 중 하나를 다시 임의로 선택해 split rule을 만들고 이를 만족하는 객체들을 노드의 왼쪽/오른쪽으로 보내는 방식을 사용한다. 그렇기 때문에 이런 split은 항상 특정 변수 축에 평행한 **axis-parallel split**이다. EIF는 random slope라는 개념을 도입해 **non-axis-parallel-split**을 도입했다.
	- [ ] 

- [ㅌ] Motivation : Standard IForest 에 대한 반례를 제시
	- [ ] 아래의 그림을 통해 3가지 경우를 살펴보자
	- [ ] Extended IForest 는 Standard IForest 가 커버하지 못하는 2번, 3번과 같은 케이스를 공략한다.
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120161717.png)

1) 첫 번째 그림은 2차원 평면에 평균이 (0,0)인 정규분포로 생성된 데이터이다. 데이터가 정규분포로 생성되었기 때문에, IF로 계산한 anomaly scores의 contour 역시 원형을 그리며 작아지는 것을 기대할 것이다. 그런데 실제 관측된 anomaly scores의 [-2, 2] 범위에선 원형의 형태를 보이지만, 그 범위를 벗어나면 직사각형의 양상을 보이고 있다. 이럴 경우 원점 (0,0)으로 부터 떨어진, 반지름은 같지만 anomaly scores가 서로 다른 결과가 발생한다. 즉, 동일한 반지름에 대해서 anomaly scores의 분산이 커지는 문제가 생긴다. 
2) 두 번째 그림은 2차원 평면에 평균이 (0,10)과 (10,0)인 정규분포 두 개로부터 생성된 데이터다. 여기선 두 개의 군집이 확인된다. Anomaly scores의 contour를 확인해보면 1번 그림과 마찬가지로 특정 범위를 넘어가면 직사각형 형태의 contour를 만들고 있다. 더 심각한 것은 (0,0)과 (10,10)을 중심으로 마치 데이터가 뭉쳐있는 것처럼 보이는 **Ghost Cluster**를 만든다는 것이다. 이 ghost cluster는 자친 정상 데이터를 anomaly로 잘못 분류해 false alarm rate을 높이고 기존 데이터에 대한 bias를 만든다는 문제를 발생시킨다. 
3) 세 번째 그림은 sin 곡선의 데이터에 noise를 준 데이터이다. 마찬가지로 직사각형 형태의 contour 문제와, 곡선 굴곡 사이의 ghost cluster를 생성한다는 문제가 발생하고 있다. 

이 그림들을 통해, IF가 계산하는 anomaly scores가 robust하지 않다는, 즉 anomaly scores의 분산이 큼을 알 수 있다. 

#### What Makes This Problem?

위와 같은 문제의 발생 원인은 IF의 split 방식에 있다. IF는 각 노드에서 임의로 선택한 변수와 그 변수의 범위에서 선택한 임의 값으로 기준을 삼아 feature space를 분할한다. 이 때 특정 축에 평행하게 제약된 상태로 split이 진행되는데, 이 axis-parallel 제약이 IF의 문제에 대한 원인이 된다. 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122220847.png)

위 그림처럼, 2차원 평면에서는 split의 방향은 가로/세로 두 가지 뿐이다. (a) 그림을 보면, 객체들이 보여있는 중심부에서 분할이 많이 일어나고 있다. 데이터가 뭉쳐져 있는 곳의 객체를 isolation시키려면 많은 split이 필요하다. 문제는 중심부에 있는 데이터를 고립시키기위해 적용한 split이 데이터가 없는 지역에 대한 split에도 영향을 준다는 것이다. 즉, axis-parallel 제약 조건이 데이터가 존재하지 않는 영역의 anomaly scores에 영향을 준다는 것이다.  
#### Contribution 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120162141.png)

> [!note]
>  But as we have seen, <font style="color:orange">the branch cuts are always either horizontal or vertical , and this introduces a bias and artifacts in the anomaly score map</font>. There is no fundamental reason in the algorithm that requires this restriction, and so at each branching point, we can select <font style="color:skyblue">a branch cut that has a random “slope”</font>.

의사결정나무는 설명력을 확보하기 위해서 축에 수직/수평인 split을 사용하는데, Standard IF는 말 자체는 trees의 ensemble인 forest 이지만, 변수에 대한 설명력을 확보하진 못한다. IF 가 변수에 대한 설명력을 확보하는 것에 대한 논문은 추후 소개할 것이다. 실질적으로 IF 알고리즘은 변수의 중요도나 영향력을 파악해주지 않는다. 그렇다면 변수값에 수직/수평인 방법으로 split을 할 필요가 없어진다. 즉, 기울기 (slope)가 있는 선으로 구분하는 것을 허용할 수 있다.

#### Illustrative example : 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120163232.png)

위 그림을 보면 Standard IF는 우측 상단의 빨간색 점을 isolation 하는데 있어서 축에 수직/수평인 split을 통해 진행하고, Extended IF는 축에 수직이 아닌 기울기가 있는 split을 통해 isolation을 한다. 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120163412.png)


#### How are the biases reduced? 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120163520.png)

분할선/기준선이 많이 겹쳐진 영역일 수록 isolation이 어려운 영역인데, 해당 영역에 대해서는 anomaly score가 낮아지고, 덜 겹쳐질수록 score가 높아진다. Standard IF와 비교해서 Extended IF는 더 정확한 isolation을 할 수 있다. 

#### Algorithm : 

EIF의 알고리즘은 IF의 Training stage에서 첫 번째 Algorithm 1은 동일하고, Algorithm 2에서 split의 방법은 동일하다. 그리고 Evaluation stage도 아주 약간의 차이가 있다. 

##### Training stage

기울기와 절편을 임의의 값을 취함으로써 Standard IF의 낮은 split 자유도를 높여줄 수 있다. 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122221949.png)
1. sampling된 $X^\prime$이 고립되었다면
	2. terminal node로 반환
3. 그렇지 않다면
	4. 방향벡터 $\vec{n}\in \boldsymbol{\mathbb{R}}^{|X|}$ 를 임의로 뽑는데, $\vec{n}$ 의 각 좌표(coordinate)는 표준정규분포에서 추출한다.
	5. $X$의 범위(range) 중 절편(intercept) 포인트 $\vec{p}\in \boldsymbol{\mathbb{R}}^{|X|}$ 를 임의로 선택한다. 
	6. extension level에 따라 $\vec{n}$의 좌표값을 0으로 조정한다
	7. $(X-\vec{p})\cdot\vec{n}\le 0$ 을 만족하는 객체들을 $X_l$ 로 보낸다.
	8. 그러지 않은 객체들은 $X_r$ 로 보낸다.
	9. 모든 데이터 포인트가 isolation될 때까지 이 $iTree()$ 과정을 반복
		- split때마다 분할 정보 $(\vec{n}, \vec{p})$ 저장
##### Evaluation stage
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122222944.png)

IF의 Algorithm 3과 차이점은 split 정보를 이용하는 방식이다. 
4.  현재 노드 $T$에 저장되어 있는 방향 벡터를 $\vec{n}$ 로 선언
5. 현재 노드 $T$에 저장되어 있는 절편 벡터를 $\vec{p}$ 로 선언
6. 데이터 포인트 $\vec{x}$ 가 $(\vec{x}-\vec{p})\cdot\vec{n}\le 0$을 만족하면 왼쪽 노드 $T.left$로 보내고, 다시 $PathLength()$ 를 수행
7. 그렇지 않다면 오른쪽 노드 $T.right$로 보내고 다시 $PathLength()$ 수행
#### Anomaly Score distribution 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240120164106.png)
위 그림 (a)와 (c)를 보면 IF가 갖는 직사각형 형태의 contour 문제가 사라졌고, ghost cluster에 대한 문제 역시 해결된 것을 알 수 있다. 

![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122223443.png)
또한 anomaly detection의 성능 지표 중 ROC curve와 PRC curve의 AUC도 EIF가 더 우수한 것을 알 수 있다. 

#### Real data

논문에서는 아래와 같은 Real data로 IF와 EIF의 성능을 비교했다 : 
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122223612.png)

이 5가지 데이터에 대한 IF와 EIF의 성능은 아래와 같다 : 
![](assets/Isolation%20Forest%20and%20Its%20Variations/Pasted%20image%2020240122223643.png)
근소하게 EIF의 성능이 IF보다 모두 우수하다는 것을 확인할 수 있다. 이는 특정 데이터에 한정된 결과이므로 항상 그렇지는 않음을 주의하자.
