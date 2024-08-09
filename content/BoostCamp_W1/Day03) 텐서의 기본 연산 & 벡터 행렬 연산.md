---
sticker: emoji//1f4d8
tags:
  - Deep_Learning
  - AI_Tech
  - BoostCamp
  - Pytorch
  - tensor
  - cat
  - expand
  - repeat
  - 산술연산
  - 비교연산
  - 논리연산
  - Norm
  - Similarity
  - 행렬곱셈
  - 대칭/상하이동
---
# 5. Basic Operations on Tensors

## 5.1 Tensor의 모양 변경2

#### `torch.cat()`

앞서 배운 `torch.stack()`은 새로운 차원을 만들어 텐서들을 결합하는 방식인 반면, `torch.cat()`은 기존 차원을 유지하면서 텐서들을 연결(concatenate)한다.

- **`torch.cat(tensors=Tuple[Tensor, ...] | List[Tensor], dim: _int=0, ...)`**
	- `dim` 인자의 default 값은 `0`으로 기본적으로 dim-0을 기준으로 텐서를 연결한다
	- 주의할 점은, `dim`으로 지정된 연결하고자 하는 **기준이 되는 차원을 제외하고는 나머지 차원들의 크기는 텐서들 간 동일해야 한다는 점**이다 
	- 기준 차원(즉, `dim`으로 지정된 차원)을 제외한 차원(들)의 크기가 다른 경우, `.reshape()`, `.view()` 등을 이용해 차원 크기를 맞춰줘야 한다

#### `.expand()` 메서드 & `.repeat()` 메서드

- **`.expand(*sizes)`** 메서드
	- 주어진 텐서의 차원 중 크기가 1인 차원에 대해 해당 차원 크기를 확장
	- 예: `f = torch.tensor([[1,2,3]])`은 shape이 `[1,3]`인 2D 텐서이다. 이를 이용해 `f.expand(4,3)` 사용하면 차원크기가 1인 dim-0 방향으로 확장이 되어 최종적으로 shape이 `[4,3]`인 2D 텐서가 된다. 
	- 예: `x = torch.tensor([[1],[2],[3]])`와 같은 `[3,1]` 2D 텐서에 `x.expand(-1,4)`를 하면 기존 `x`의 첫 번째 차원(dim-0)을 유지하면서 shape이 `[3,4]`인 2D 텐서가 된다.

```python
x = torch.tensor([[1],[2],[3]])
x.expand(-1, 4)
```

```
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
```

- **`.repeat(*sizes)`** 메서드:
	- `.expand()` 메서드처럼 텐서의 요소들을 반복해서 차원 크기를 확장하는 메서드이다. 
	- 다만, `.expand()` 메서드처럼 텐서의 차원 중 일부의 크기가 1이어야 하는 제약이 없다(장점)
		- 단점은, `.repeat()` 메서드는 실제 복사본을 만들어 추가적인 메모리를 할당하기 때문에, 가상 view를 생성하여 메모리를 할당하지 않는 `.expand()` 메서드보다는 메모리 효율성이 떨어진다
	- 예: `h = torch.tensor([[1,2],[3,4]])`인 2D 텐서에 `h.repeat(2,3)`를 사용하면 dim-0 축으로 2번, dim-1 축으로 3번 반복해 텐서 크기를 확장하는 것이다. 

```python
h = torch. tensor([[1, 2],
                   [3, 4]])

i = h.repeat(2, 3)

print('i =' , i)
print('i.shape =', i.shape)

print('--------------------------\n')
i = h.repeat(2, 3, 3)

print('i =' , i)
print('i.shape =', i.shape)
```

```
i = tensor([[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]])
i.shape = torch.Size([4, 6])
--------------------------

i = tensor([[[1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4],
         [1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4],
         [1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4]],

        [[1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4],
         [1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4],
         [1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4]]])
i.shape = torch.Size([2, 6, 6])
```

## 5.2 Tensor의 기초 연산

#### 산술 연산

- **더하기**: 
	- **`torch.add(input: Number | Tensor | _complex, other: Number | Tensor | _complex, ...)`**: Broadcasting 지원
		- broadcasting: 작은 텐서를 큰 텐서와 같은 shape으로 확장해 계산
	- in-place: **`.add_(other: Number | Tensor | _complex | SymInt | SymFloat, ...)`**
		- in-place 방식은 메모리를 절약하여 텐서를 업데이트할 수 있는 연산 방식
			- 추가적인 메모리 할당이 필요 없기 때문에 메모리 측면에서 장점이 있지만, _autograd와의 호환성 문제가 발생할 위험이 있기 때문에_ 신중히 사용해야 한다. 
- **빼기**: 
	- **`torch.sub(input: Number | Tensor | _complex, other: Number | Tensor | _complex, ...)`**: 
		- Broadcasting 지원
	- in-place: **`.sub_(other: Number | Tensor | _complex | SymInt | SymFloat, ...)`**
	  
- **곱하기**: 
	- **스칼라 곱**: 스칼라 `k`와 텐서 `a`가 주어졌을 때, `torch.mul(k, a)`
		- in-place: `a.mul_(k)`
	- **요소별 곱(Hadamard/Element-wise product)**: 
		- **`torch.mul(input: Number | Tensor | _complex, other: Number | Tensor | _complex,...)`**:
			- Broadcasting 지
		- in-place: **`.mul_(other: Number | Tensor | _complex | SymInt | SymFloat)`**

- (요소별) **나누기**: 
	- **`torch.div(input: Number | Tensor, other: Number | Tensor, ...)`**: 
		- Broadcasting 지원
	- in-place: **`.div_(other: Number | Tensor, ...)`**
		- 주의할 점: in-place 방식을 사용할 때는 자료형에 주의해야 한다(float형 사용 권장)

- (요소별) **제곱**: 
	- **`torch.pow(input: Tensor, exponent: Number, ...)`**: 
		- 주어진 텐서의 각 요소들에 대해 `exponent`에 주어진 수만큼 제곱승
	- **`torch.pow(input: Tensor, exponent: Tensor, ...)`**: 
		- 주어진 텐서 `input`의 각 요소를 `exponent` 텐서의 각 요소로 제곱승
	- in-place: **`.pow_(exponent: Tensor | Number)`**

- (요소별) **거듭제곱근**: 
	- 마찬가지로 **`torch.pow()`** 이용
	- 예: 텐서 `s`의 각 요소들의 n 제곱근 계산
		- `torch.pow(s, 1/n)`
		- in-place: `s.float().pow_(1/n)`

#### 비교 연산

- **Equal**: **`torch.eq(input: Tensor, other: Tensor, ...)`**
	- 두 텐서의 모든 요소가 동일한지 여부 반환
	- Boolean 텐서 반환
- **Not Equal**: **`torch.ne(input: Tensor, other: Tensor, ...)`**
	- 두 텐서의 모든 요소가 다른지 여부 반환
	- Boolean 텐서 반환
- **Greater than**: **`torch.gt()**
- **Greater or Equal**: **`torch.ge()`**
- **Less than**: **`torch.lt()**
- **Less or Equal**: **`torch.le()`**

#### 논리 연산

- **논리곱(AND)**: $x\land y$ 
	- 입력 $x$와 $y$가 _모두 참_ 일 때 출력이 _참_
	- **`torch.logical_and(input: Tensor, other: Tensor, ...)`**
- **논리합(OR)**: $x\lor y$ 
	- 입력 $x$와 $y$ 중 _하나라도 참_ 이면 출력이 _참_
	- **`torch.logical_or(input: Tensor, other: Tensor, ...)`**
- **배타적 논리합(XOR)**: $x\oplus y$ 
	- 입력 $x$와 $y$ 중 _단 하나만 참_ 일 때, 출력이 _참_
	- **`torch.logical_xor(input: Tensor, other: Tensor, ...)`**

| $x$ | $y$ | $x\land y$ | $x\lor y$ | $x\oplus y$ | 
| --- | --- | ---------- | --------- | ----------- |
| T   | T   | T          | T         | F           |
| T   | F   | F          | T         | T           |
| F   | T   | F          | T         | T           |
| F   | F   | F          | F         | F           |

---

# 6. Operations on Tensor: Vectors & Matrices

## 6.1 Tensor의 노름

- **1D 텐서의 노름**: 해당 벡터가 **원점에서 얼마나 떨어져 있는가**를 의미
	- 벡터의 길이를 측정
- $L_p$-norm: 크기가 $n$인 1D 텐서 $\boldsymbol{x}$와 $0<p\le\infty$에 대하여 $$||\boldsymbol{x}||_p=\left(\sum_{i=1}^n|x_i|^p\right)^{\frac{1}{p}}$$
- **L1 노름**: 1D 텐서 요소들의 절대값의 합 $$||\boldsymbol{x}||_1=\sum_{i=1}^n|x_i|$$
	- _맨해튼 노름(Manhattan norm)_ 이라고도 함
	- **`torch.norm(x, p=1)`**


- **L2 노름**: 1D 텐서 요소들의 제곱합의 제곱근 $$|\boldsymbol{x}||_2=\sqrt{\sum_{i=1}^n|x_i|^2}$$
	- _유클리드 노름_ 이라고도 함
	- **`torch.norm(x, p=2)`**

- **$L_\infty$ 노름**: 1D 텐서 요소들의 절대값 중 최대값 $$||\boldsymbol{x}||_\infty=\max(|x_1|,\ldots,|x_n|)$$
	- **`torch.norm(x, p=float('inf'))`** or **`torch.max(x.abs())`**

![[Pasted image 20240807152442.png|700]]출처: https://www.youtube.com/watch?v=NKuLYRui-NU


## 6.2 유사도(Similarity)

- **유사도(similarity)**: 두 벡터가 얼마나 유사한지 측정한 값
	- 여기서는 _거리 기반_ 유사도 측정 방법을 다룬다.

- **맨해튼 유사도**: 두 벡터 간 _L1 노름_ / _맨해튼 거리_ 에 역수를 취해 계산
	- 0 ~ 1의 값을 가지며, 1에 가까울수록 두 벡터가 유사한 것
	- 크기가 $n$인 두 벡터 $\boldsymbol{x}$, $\boldsymbol{y}$에 대해 $$\begin{aligned}\text{Manhattan Distance}&=\sum_{i=1}^n|x_i-y_i|\\\text{Manhattan Similarity} &= \frac{1}{1+\text{Manhattan Distance}}\in \mathbb{R}^{[0,1]}\end{aligned}$$
	- 맨해튼 거리: **`manhattan_distance = torch.norm(x-y, p=1)`**
	- 맨해튼 유사도: **`1/(1 + manhattan_distance)`**

- **유클리드 유사도**: 두 벡터 간 _유클리드 거리_ 에 역수를 취해 계산
	- 0 ~ 1의 값을 가지며, 1에 가까울수록 두 벡터가 유사한 것
	- 크기가 $n$인 두 벡터 $\boldsymbol{x}$, $\boldsymbol{y}$에 대해 $$\begin{aligned}\text{Euclidean Distance}&=\sqrt{\sum_{i=1}^n|x_i-y_i|^2}\\\text{Euclidean Similarity} &= \frac{1}{1+\text{Euclidean Distance}}\in \mathbb{R}^{[0,1]}\end{aligned}$$
	- 유클리드 거리: **`euclidean_distance = torch.norm(x-y, p=2)`**
	- 유클리드 유사도: **`1/(1 + euclidean_distance)`**

- **코사인 유사도**: 두 벡터간 각도의 코사인 값을 계산
	- -1 ~ 1의 값을 가지며, 1에 가까울수록 두 벡터가 유사한 것
	- 각도 측정: 두 벡터간 내적 이용
	- 크기가 $n$인 두 벡터 $\boldsymbol{x}$, $\boldsymbol{y}$에 대해 $$\cos(\boldsymbol{x},\boldsymbol{y}) = \frac{\langle \boldsymbol{x},\boldsymbol{y}\rangle}{||\boldsymbol{x}||_2||\boldsymbol{y}||_2}=\frac{\sum_i x_iy_i}{||\boldsymbol{x}||_2||\boldsymbol{y}||_2}$$
	- 코사인 유사도: **`cosine_similarity = torch.dot(x,y) / (torch.norm(x,p=2) * torch.norm(y,p=2))`**

## 6.3 2D Tensor: 행렬의 곱셈 연산

- 두 개의 2D 텐서 $A$와 $B$, 즉 두 행렬의 곱셈은 행렬 $A$의 $i$번째 행벡터와 행렬 $B$의 $j$번째 열벡터 간 내적을 각 성분으로 갖는 행렬, 2D 텐서이다.
	- 행렬 곱 $AB$에 대한 또 다른 해석: 
		- 행렬 $A$의 열벡터의 선형 결합
		- 행렬 $B$의 행벡터의 선형 결합

- 행렬 곱셈 코드: 2D 텐서 `A`와 `B`가 있다고 하자
	- **`A.matmul(B)`**: 1D 또는 다차원 텐서에서도 사용 가능
		- Broadcasting 지원
	- **`A @ B`**: 다차원 텐서에 사용 가능
		- Broadcasting 지원
	- **`A.mm(B)`**: 두 텐서가 반드시 2D 텐서이어야 함. 다차원 텐서에서 사용 불가
		- _Broadcasting 지원하지 않음_
		- 제약 사항이 많아보이지만 덕분에 _디버깅_ 에 유리하다


#### 흑백 이미지 대칭 이동

아래의 같은 2D 텐서가 주어졌다고 하자.

```python
G = torch.tensor([[255, 114, 140],
				  [39, 255, 46],
				  [61, 29, 255]])
import matplotlib.pyplot as plt

plt.xticks([]), plt.yticks([])
plt.imshow(G, cmap='gray', vmin=0, vmax=255)
plt.show()
```

![[Pasted image 20240807155244.png]]

- 이 흑백 이미지 `G`를 좌우로 뒤집는 대칭 이동은 행렬 `G`에 아래 행렬을 곱하면 된다. $$\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix}$$
```python
H = torch.tensor([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])
I = G @ H
plt.xticks([]), plt.yticks([])
plt.imshow(I, cmap='gray', vmin=0, vmax=255)
plt.show()
```

![[Pasted image 20240807155405.png]]

#### 흑백 이미지 상하이동

- 비슷하게 이미지를 상하 방향으로 대칭이동 시키려면 다음과 같이 생각할 수 있다. 
	- 상하 대칭이동 행렬 $H$를 찾고자 함

$$
\begin{aligned}
G\times H &= I\\
&\Longleftrightarrow H = G^{-1}I
\end{aligned}
$$

- $G$의 역행렬을 직접 계산하기 위해선, `torch.linalg.inv` 이용
	- 그러나 이 방법은 아이러니하게 상하 대칭이동된 결과 행렬 $I$를 알아야 함: 기각

- _다른 방법_ : 앞서 구한 $\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix}$를 $G$의 왼쪽에 곱하기

```python
I = H @ G

print('I = ', I)
```

```
I =  tensor([[ 61,  29, 255],
        [ 39, 255,  46],
        [255, 114, 140]])
```

```python
plt.xticks([]), plt.yticks([])
plt.imshow(I, cmap='gray', vmin=0, vmax=255)
plt.show()
```

![[Pasted image 20240808130607.png]]

- _간편한 방법_ : 
	- 좌우 대칭: **`torch.fliplr()`**
	- 상하 대칭: **`torch.flipud()`**

```python
plt.xticks([]), plt.yticks([])
plt.imshow(torch.fliplr(G), cmap='gray', vmin=0, vmax=255)
plt.show()
```

![[Pasted image 20240807155940.png]]

```python
plt.xticks([]), plt.yticks([])
plt.imshow(torch.flipud(G), cmap='gray', vmin=0, vmax=255)
plt.show()
```

![[Pasted image 20240807160026.png]]


---
