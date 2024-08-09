---
sticker: emoji//1f4d8
tags:
  - Deep_Learning
  - AI_Tech
  - BoostCamp
  - Pytorch
  - tensor
  - 초기화
  - CUDA
  - indexing
  - slicing
  - view
  - reshape
  - flatten
  - transpose
  - squeeze/unsqueeze
  - stack
---
# 3. Creating Tensors

## 3.1 Tensor 생성

텐서를 생성하는 데에는 여러 방법이 있는데 하나씩 살펴본다.

### **특정 값으로 초기화**하여 생성 및 변환

특정한 값으로 초기화하여 텐서를 생성하는 것은 크게 두 가지 방법이 있다.

- **`torch.zeros(size: Sequence[_int | SymInt], dtype: _dtype, ...)`**: 
	- 0으로 채워진 tensor 생성
	- 첫 번째 인자 `size`:
		- 정수형 scalar가 들어가면 0으로 채워진 1D 텐서 반환: `torch.zeros(5)`
		 - 정수형 `list`가 들어가면 0으로 채워진 $N$-D 텐서 반환: `torch.zeros([3,2])`
	- `dtype`: 텐서의 데이터 타입 지정
		- default: `torch.set_default_dtype()`으로 지정된 타입으로 지정됨
			- 기본적으로 `torch.get_default_dtype()`을 확인해 보면 `float32`이므로 명시적으로 지정하지 않을 시 `float32`라고 이해하자.
- **`torch.ones(size: Sequence[_int | SymInt], dtype: _dtype, ...)`**: 
	- 1로 채워진 텐서 생성
	- 각 인자에 대한 내용은 `torch.zeros()`와 동일함

- 어떤 텐서의 크기와 자료형과 같게 0 또는 1로 초기화된 텐서를 만드는 방법: **`*_like()`**
	- **`torch.zeros_like()`**: 
		- `torch.zeros_like(a)`: 텐서 `a`와 크기(shape)와 자료형(`dtype`)이 동일한 **0**으로 채워진 텐서 생성
	- **`torch.ones_like()`**
		- `torch.ones_like(a)`: 텐서 `a`와 크기(shape)와 자료형(`dtype`)이 동일한 **1**로 채워진 텐서 생성

### **난수로 초기화**하여 생성 및 변환

랜덤한 난수로 초기화된 값으로 텐서를 생성하는 대표적인 두 방법이 있다.

- **`torch.rand(size: Sequence[_int | SymInt], dtype: _dtype, ...)`**: 
	- 연속균등분포 $Unif(0,1)$에서 텐서 생성
	- $Unif(-3,3)$에서 만들고자 하면: `6 * torch.rand(100) - 3`
	- `size`: 
		- 정수형 scalar가 들어가면 0으로 채워진 1D 텐서 반환: `torch.rand(5)`
		 - 정수형 `list`가 들어가면 0으로 채워진 $N$-D 텐서 반환: `torch.rand([3,2])`
	- `dtype`: 
		- default: `torch.set_default_dtype()`으로 지정한 값
			- `torch.get_default_dtype()`은 기본적으로 `float32`
- **`torch.randn(size, dtype, ...)`**: 
	- 표준정규분포 $\mathcal{N}(0,1)$에서 텐서 생성
	- $\mathcal{N}(\mu,\sigma^2)$에서 만들고자 하면: `mu + torch.randn(100) * sigma`
	- `size`와 `dtype`에 대해선 `torch.rand()`와 동일

> [!warning] 
> `torch.rand()`와 `torch.randn()`에 `dtype`에 `torch.int`로 지정하면 에러가 난다. 이는 $Unif(0,1)$과 $\mathcal{N}(0,1)$의 support가 모든 실수 $\mathbb{R}$이기 때문이다. 

- 특정 텐서와 크기 및 자료형이 같은 난수 텐서 생성: **`*_like()`**
	- **`torch.rand_like()`**:
		- `torch.rand_like(a)`: 텐서 `a`와 크기와 자료형이 같으면서 $Unif(0,1)$의 난수로 채워진 텐서 생성
	- **`torch.randn_like()`**:
		- `torch.randn_like(a)`: 텐서 `a`와 크기와 자료형이 같으면서 $\mathcal{N}(0,1)$의 난수로 채워진 텐서 생성

### **특정 범위로 초기화**하여 생성

- 특정한 범위 내에서 일정한 간격(step)으로 초기화되는 텐서 생성 방법: `numpy.arange()`와 같은 방식
- **`torch.arange(start: Number, end: Number, step: Number, ...)`**: 
	- `start`부터 `end - step`까지 `step` 간격의 텐서 생성
	- `start`:
		- default: `0`
	- `end`: 
		- `start`와 `step`을 기입하지 않고 `end`만 기입 시:
			- `end`가 정수인 경우: 예) `torch.arange(3)`
				- 정수형 텐서 반환
			- `end`가 실수인 경우: 예) `torch.arange(5.5)`
				- 실수형 텐서 반환 `tensor([0, 1, 2, 3, 4, 5])`
	- `step`: 
		- default: `1`
	- `torch.arange()`의 `dtype`은 `start`, `end`, `step` 중 어느 하나라도 실수형 값을 갖는 경우에는 `float`형, 그렇지 않으면 `int`형

### **초기화 하지 않은** Tensor 생성

- _초기화 하지 않음_ : 생성될 텐서의 각 요소를 명시적으로(explicitly) 특정 값으로 지정하지 않았음을 의미
	- 초기화 되지 않은 텐서 생성 시, 각 요소는 메모리에 존재하는 값들 중 임의로 채워짐
- 이러한 텐서 생성 방식을 사용하는 **이유**: 
	- _성능 향상_ :
		- 텐서가 생성 후 어느 곳에도 이용되지 않고 다른 값으로 대체될 경우, 특정 값을 할당함으로 초기화하는 것은 자원 소모
	- _메모리 효율성 증대_ 

- **`torch.empty(size: Sequence[_int | SymInt], ...)`**: 
	- 주어진 `size` 에 따라 초기화 되지 않은 텐서 생성
	- `size`는 scalar 값일 수도, list 값일 수도 있음
		- `size`로 shape 결정
	- **`.fill_(value)`** 메서드를 통해 초기화 되지 않은 텐서에 주어진 값(`value`)으로  채울 수 있음
		 - `.fill_()` 메서드는 inplace 방식으로 동작하여 메모리 주소가 변하지 않음.

```python
q = torch.empty(5)
print('q = ', q)
before_id = id(q)
torch.fill_(q, 3.0)
print('q = ', q)
after_id = id(q)
print('메모리 주소 동일한지 여부:', before_id==after_id)
```

```plaintext
q = tensor([2.7509e+35, 4.4298e-41, 2.7509e+35, 4.4298e-41, 0.0000e+00]) q = tensor([3., 3., 3., 3., 3.]) 메모리 주소 동일한지 여부: True
```

### `list`, `numpy.ndarray`로부터 생성

대괄호(square bracket) `[]` 내에서 쉼표(`,`)로 구분해 생성된 `list`를 텐서로 만들려면 해당 `list` 객체를 `torch.tensor()`에 입력하기만 하면 된다. 

- 이때 `list`의 각 요소에 대한 자료형은 유지된다. 
	- `torch.tensor()`의 `dtype` 인자를 통해 명시적으로 변경할 수도 있

```python
s = [1, 2, 3, 4, 5, 6]
t = torch.tensor(s)

print('t =', t)
print(t.dtype)
```

```
t = tensor([1, 2, 3, 4, 5, 6])
torch.int64
```

```python
s = [1., 2., 3., 4., 5., 6.]
print(torch.tensor(s).dtype)
print(torch.tensor(s, dtype=int).dtype)
```

```
torch.float32
torch.int32
```

- `numpy.array()`를 통해 생성된 `numpy.ndarray` 객체는 **`torch.from_numpy()`** 를 통해 텐서로 변환
	- Numpy로 생성된 텐서는 기본적으로 정수형이므로 실수형으로 타입 캐스팅이 필요함
		- 엄밀히 말하면 `ndarray` 객체의 자료형을 따라간다. 예를 들어 `np.array()`로 선언 시 `dtype=np.float32`로 지정하면 `torch.from_numpy()`를 사용 시 `dtype`은 `torch.float32`가 된다. 
		- 하지만 일반적으로 이렇게 디테일하게 하지 않으므로, `v = torch.from_numpy(u).float()`처럼 타입 캐스팅을 하는 것이 권장됨

### CPU Tensor

텐서에는 크게 CPU 텐서와 CUDA 텐서가 있는데, 우선 CPU 텐서에 대해 살펴본다. 

- 정수형(`int32`) CPU 텐서: **`torch.IntTensor()`**

```python
w = torch.IntTensor([1, 2, 3, 4, 5])
print('w =', w)
print('w.device =', w.device)
```

```
w = tensor([1, 2, 3, 4, 5], dtype=torch.int32)
w.device = cpu
```

- 실수형(`float32`) CPU 텐서: **`torch.FloatTensor()`**

```python
A, B, C, D, E = 1, 2, 3, 4, 5
x = torch.FloatTensor([A, B ,C, D, E])

print('x =', x)
print('x.dtype =', x.dtype)
print('x.device =', x.device)
```

```
x = tensor([1., 2., 3., 4., 5.])
x.dtype = torch.float32
x.device = cpu
```

- 기타: 
	- 정수형 CPU 텐서: 
		- `torch.ByteTensor()`: `uint8`
		- `torch.CharTensor()`: `int8`
		- `torch.ShortTensor()`: `int16`
		- `torch.LongTensor()`: `int64`
	- 실수형 CPU 텐서: 
		- `torch.DoubleTensor()`: `float64`

### Tensor 복제

텐서를 복제하는 방법에는 두 가지가 있다.

- **`.clone()`** 메서드:
	- 텐서의 복사본을 만들되, 원본과는 별도의 메모리를 사용해 독립적으로 새로운 존재
- **`.detach()`** 메서드: 
	- 텐서를 _연산그래프에서 분리_ 하여, 원본과 데이터를 공유하지만, gradient 계산을 수행하진 않음

### CUDA Tensor

- GPU를 사용하는 이유: 
	- 수많은 작은 코어로 인한 **효율적인 병렬 처리**가 가능하여 대량 연산을 동시에 수행함
	- 이러한 병렬 처리 능력은 AI 모델의 훈련 및 추론 속도를 크게 향상시킴
	- 경제적 이점

일반적으로 `torch.tensor()`로 텐서를 생성한 후 현재 환경에서 GPU가 사용 가능한지 확인 후 텐서의 `.to()` 메서드 또는 `.cuda()` 메서드를 사용해 CUPA 텐서로 만든다. 

```python
a = torch.tensor([1, 2, 3])
print('a.device =', a.device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

a = a.to(device) # a.cuda()
print('a.device =', a.device)
```

```
a.device = cpu
device = cuda
a.device = cuda:0
```

- `torch.cuda.is_available()`: 현재 환경이 CUDA를 사용할 수 있는지 확인
	- boolean 반환
- `torch.cuda.get_device_name(device=0)`: 현재 환경의 첫 번째 CUDA device 이름 확인
- `torch.cuda.device_count()`: 현재 환경에서 사용 가능한 CUDA device 개수
- CUDA 텐서를 `.to('cpu')`를 통해 다시 CPU 텐서로 만들 수 있음

---

# 4. Manipulation of Tensors

## 4.1 Indexing & Slicing

### Indexing과 Slicing

텐서의 indexing과 slicing은 Numpy의 것과 동일하다고 생각하면 된다. 이미 잘 알고 있는 내용이므로 자세히 다루진 않는다. 

- 특정 차원(dim-k)의 길이를 `len(k)`라고 할 때 해당 dim-k 차원에서 특정 인덱스 `idx`의 음수 인덱스: `idx - len(k)`
- slicing 시, 각 차원의 전체를 의미하는 것은 `:` 이외에도 `...`가 있다. 

## 4.2 모양 변경 1

### **`view()`** 메서드 & **`reshape()`** 메서드

- `.view()`와 `.reshape()` 모두 주어진 텐서를 원하는 shape으로 변경하는 메서드이다. 
	- `.view()`는 주어진 텐서가 메모리 레이아웃에서 **연속적(contiguous)** 해야 하는 반면, 
	- `.reshape()`은 이와는 관계 없이 새로운 복사본을 할당하여 모양을 변경
	- 안전성 및 유연성이 중요할 때는 `.reshape()`을 사용하고, 메모리의 연속성이 확실하고 성능이 중요할 경우 `.view()`를 사용하자. 

#### `.view() `메서드 관련

_4강 퀴즈_ 를 풀던 도중 비연속적(non-contiguous)인 경우에도 `view()` 메서드가 동작하는 경우가 있음을 알게 되었고, 궁금증이 생겨 찾아본 내용이다. 

- `.view()` 메서드는 일반적으로 tensor가 메모리에서 연속적(contiguous)한 경우에 사용 가능
    - 단, tensor가 전체적으로 비연속적(non-contiguous)이더라도, 특정한 하나의 축/차원에 대해서 연속적이기만 하면 오류 나지 않고 동작하되, `reshape()` 처럼 동작함
    - 즉, 새로운 객체로 저장됨

### 예시

```python
t = torch.tensor([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]])
print(t)
print(t[:2, :2])
print(t[:2,:2].is_contiguous())
t[:2,:2].view(1,-1)
```

```
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
tensor([[1, 2],
        [5, 6]])
False
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-138-7c037886a398> in <cell line: 8>()
      6 print(t[:2, :2])
      7 print(t[:2,:2].is_contiguous())
----> 8 t[:2,:2].view(1,-1)

RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

위와 같이 `t[:2,:2]` 텐서의 `[1,2]` 와 `[5,6]` 은 `t` 에서 비연속적이기 때문에 `view()` 적용 시 에러가 난다. 

```python
print(t[:2, :1])
print(t[:2,:1].is_contiguous())
t[:2,:1].view(1,-1)
```

```
tensor([[1],
        [5]])
False
tensor([[1, 5]])
```

반면에 `t[:2,:1]` 텐서의 `[1]` 와 `[5]` 는 원본 `t` 텐서에서 비연속적이지만, 첫 번째 차원(dim-0) 기준으로는 메모리 레이아웃에서 연속적, 즉 행방향(dim-0 방향)으로 연속적으로 존재하기 때문에 `view()`를 사용하더라도 오류가 나지 않는다.

다른 예시를 들어보자면, 마찬가지로 원본 텐서에 대해서는 비연속적이지만, 특정한 하나의 축에 대해 연속적인 경우를 보기 위해 shape이 `[2,3,4]` 인 텐서를 보자.

```python
t = torch.arange(1, 25).view(2, 3, 4)
print("Original Tensor:\\n", t)

selected = t[:1, :, :2]
print("\\nSelected Sub-Tensor:\\n", selected)
print("Is contiguous?:", selected.is_contiguous())
print("View operation result:", selected.view(-1))
```

```
Original Tensor:
 tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],

        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]])

Selected Sub-Tensor:
 tensor([[[ 1,  2],
         [ 5,  6],
         [ 9, 10]]])
Is contiguous?: False
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-165-128c53bdbbc2> in <cell line: 7>()
      5 print("\\nSelected Sub-Tensor:\\n", selected)
      6 print("Is contiguous?:", selected.is_contiguous())
----> 7 print("View operation result:", selected.view(-1))

RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

이 경우 `t[0,:,:2]`도 마찬가지로 `[1,2]`, `[5,6]`, `[9,10]`은 비연속적이기 때문에 `view()`가 동작하지 않는는다.

```python
t = torch.arange(1, 25).view(2, 3, 4)
print("Original Tensor:\\n", t)

selected = t[:1, :, :1]
print("\\nSelected Sub-Tensor:\\n", selected)
print("Is contiguous?:", selected.is_contiguous())
print("View operation result:", selected.view(-1))
```

```
Original Tensor:
 tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],

        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]])

Selected Sub-Tensor:
 tensor([[[1],
         [5],
         [9]]])
Is contiguous?: False
View operation result: tensor([1, 5, 9])
```

위 경우는 `t[:1,:,:1]`이 비연속적이더라도 두 번째 차원(dim-1) 기준으로 연속적이기 때문에 `view()`가 에러 나지 않고 동작함을 볼 수 있다.

- 결론:
    - `view()` 메서드는 비연속적인 경우에, 특정한 하나의 축/차원에 대해서 연속적이라면 에러 나지 않는다.
        - 하지만, 이런 경우는 `[[1],[5]]` 이나 `[[[1],[5],[7]]]` 처럼 하나의 축을 제외한 나머지 차원의 길이가 1인 경우에만 해당되므로 그다지 쓸모 있는지는 모르겠음
    - 그러니 비연속적이면 `.contiguous()` 를 쓰거나, 안전하게 `.reshape()`을 쓰는 것이 좋을 듯!

### **`flatten()`**: 텐서 평탄화

- **`flatten(input: Tensor, start_dim, end_dim, ...)`**:
	- 주어진 텐서를 1차원 텐서로 만들거나, 평탄화할 차원을 지정하여 shape을 변경할 수 있는 함수
	- `start_dim`(default: `0`) 과 `end_dim`(default: `-1`)이 지정되지 않으면 주어진 텐서 `input`을 1차원 텐서로 변환

- shape이 `[3, 2, 2]`인 텐서 `k`에 대해: 
	- `torch.flatten(k)`: shape이 `[12]`인 1D 텐서 반환
	- `torch.flatten(k,1)`: shape이 `[3,4]`인 2D 텐서 반환
	- `torch.flatten(k, 0, 1)`: shape이 `[6,2]`인 2D 텐서 반환
		- `start_dim`과 `end_dim`을 지정할 바엔 `.view()`나 `.reshape()`을 쓰는 것이 더 이해하기 편할 듯?

### **`.transpose()`** 메서드: 특정 두 차원 축 전치

- **`.transpose(dim0, dim1)`**: 
	- 주어진 텐서의 차원/축들 중 두 개를 지정하여 전치하는 메서드
- 예: shape이 `[3,3,2]`인 텐서의 두 번째(dim-1), 세 번째(dim-2)를 전치
	- 여기서 `s.transpose(1,2)`나 `s.transpose(2,1)`이나 동일함

```python
s = torch.tensor([[[0, 1],
                  [2, 3],
                  [4, 5]],

                 [[6, 7],
                  [8, 9],
                  [10, 11]],

                 [[12, 13],
                  [14, 15],
                  [16, 17]]])

t = s.transpose(1, 2)
print('t = ', t)
```

```
t =  tensor([[[ 0,  2,  4],
         [ 1,  3,  5]],

        [[ 6,  8, 10],
         [ 7,  9, 11]],

        [[12, 14, 16],
         [13, 15, 17]]])
```


### 텐서의 차원 축소 및 확장: **`squeeze()`** & **`unsqueeze()`**

- **`torch.squeeze()`**: 
	- `dim` 인자가 지정되지 않을 시, 주어진 텐서에 크기가 1인 차원을 모두 축소
		- 예: shape이 `[1,1,1,4]`인 텐서 `a`는 `torch.squeeze(a)`를 통해 shape이 `[4]`가 됨
	- `dim` 인자에는 scalar 정수 또는 list가 들어갈 수 있는데, `dim`에 주어진 차원 중 크기가 1인 차원을 축소
		- 예: shape이 `[1,2,4]`인 텐서 `a`는 `torch.squeeze(a, dim=0)`를 통해 shape이 `[2,4]`가 됨
			- 이때, `torch.squeeze(a, dim=[0,1])`를 한다고 해도 크기가 1인 차원은 dim-0 뿐이므로 shape은 `[2,4]`가 됨
- **`torch.unsqueeze()`**: 
	- 반드시 `dim` 인자 값이 지정되어야 함
	- 주어진 텐서를 지정된 `dim` 인자 값에 해당하는 차원 방향으로 확장
	- `dim`에 음의 정수가 올 수도 있는데, 인덱스와 유사하게 간주하면 된다.
		- 예: shape이 `[3,4]`인 텐서 `a`가 존재
			- `dim=0` 또는 `dim=-3`: `torch.unsqueeze(a, dim=0)`의 shape → `[1,3,4]`
			- `dim=1` 또는 `dim=-2`: `torch.unsqueeze(a, dim=1)`의 shape → `[3,1,4]`
			- `dim=2` 또는 `dim=-1`: `torch.unsqueeze(a, dim=2)`의 shape → `[3,4,1]`

### **`stack()`**: 텐서 결합

- **`torch.stack(tensors: Tuple|List, dim: _int)`**
	- 합치고자 하는 텐서들을 튜플 또는 리스트 형식으로  첫 번째 인자로 사용
	- `dim`: 
		- default: `0`
		- 텐서를 합쳐 확장되는 차원을 지정
		- 음의 정수 가능: 이 경우 인덱스처럼 생각하면 됨

```python
red_channel = torch.tensor([[255, 0],
                            [0, 255]])
green_channel = torch.tensor([[0, 255],
                              [0, 255]])
blue_channel = torch.tensor([[0, 0],
                             [255, 0]])

# 첫 번째 차원 dim-0 방향으로 결합
a = torch.stack((red_channel, green_channel, blue_channel))
print('a = ', a)
print('a.shape = ', a.shape)
print('----------------------------------')

# 두 번째 차원 dim-1 방향으로 결합
a = torch.stack((red_channel, green_channel, blue_channel), dim = 1)
print('a = ', a)
print('a.shape = ', a.shape)
print('----------------------------------')


# 세 번째 차원 dim-3 방향으로 결합
a = torch.stack((red_channel, green_channel, blue_channel), dim = 2)

print('a = ', a)
print('a.shape = ', a.shape)
```

```
a =  tensor([[[255,   0],
         [  0, 255]],

        [[  0, 255],
         [  0, 255]],

        [[  0,   0],
         [255,   0]]])
a.shape =  torch.Size([3, 2, 2])
----------------------------------
a =  tensor([[[255,   0],
         [  0, 255],
         [  0,   0]],

        [[  0, 255],
         [  0, 255],
         [255,   0]]])
a.shape =  torch.Size([2, 3, 2])
----------------------------------
a =  tensor([[[255,   0,   0],
         [  0, 255,   0]],

        [[  0,   0, 255],
         [255, 255,   0]]])
a.shape =  torch.Size([2, 2, 3])
```

---
