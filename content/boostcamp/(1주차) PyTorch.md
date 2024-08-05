---
tags:
  - Deep_Learning
  - AI_Tech
  - BoostCamp
  - Pytorch
  - Lv1
---

# 1. PyTorch Intro

## 1.1 PyTorch

- PyTorch는 _간편한 딥러닝 API_를 제공하며, ML 알고리즘을 구현하고 실행하기 위한 _확장성이 뛰어난 멀티플랫폼_ 프로그래밍 인터페이스
	- Raschka, Liu & Mirjalili, 2022

- **동적 계산 그래프(Dynamic Computational Graph)** - Define by Run
	- PyTorch는 _연산을 평가_하고, _계산을 실행_하고, _구체적인 값을 즉시 반환_하는 명령형 프로그래밍 환경을 제공함
		- Raschka, Liu & Mirjalili, 2022

## 1.2 Tensor

#### Tensor란?

- PyTorch의 핵심 데이터 구조
	- Numpy의 다차원 배열(`ndarray`)과 유사한 형태
- 하나의 tensor는 여러 가지로 표현할 수 있다: 언어적/대수적/공간으로/코드로 표현

#### 0-D Tensor = Scalar

- 언어적 표현: 하나의 숫자로 표현되는 양(quantity)
- 대수적 표현: $$a = a_1,\;a\in\mathbb{R}$$
- 공간에서 표현:

![[Pasted image 20240805135135.png]]

- 코드로 표현: 

```python
import torch
a = torch.tensor(36.5)
print(f'a = {a}')
print('a =', a)
```

```plaintext
a = 36.5
a = tensor(36.5000)
```

#### 1-D Tensor = Vector

- 언어적 표현: 순서가 지정된 여러 숫자가 일렬로 나열된 구조
- 대수적 표현: $$b = (b_1,\ldots,b_n),\;b\in\mathbb{R}^n$$
- 공간에서 표현: 

![[Pasted image 20240805140718.png|600]]
- 코드 표현: 

```python
b = torch.tensor([175, 60, 81, 0.8, 0.9, 1.2])
print('b = ', b)
```

```plaintext
b =  tensor([175.0000,  60.0000,  81.0000,   0.8000,   0.9000,   1.2000])
```

#### 2-D Tensor = Matrix

- 언어적 표현: _동일한 크기_를 가진 1D tensor들이 모여서 형성한 행렬
- 대수적 표현: $$C=\begin{pmatrix}c_{11}&c_{12}&\cdots&c_{1n}\\c_{21}&c_{22}&\cdots&c_{2n}\\\vdots&\vdots&\ddots&\vdots\\c_{m1}&c_{m2}&\cdots&c_{mn}\end{pmatrix},\;C\in\mathbb{R}^{m\times n}$$
- 공간 표현: $3\times5$ 행렬

![[Pasted image 20240805141539.png]]


- 코드 표현: 위 $3\times5$ 행렬을 grayscale 흑백 이미지로 표현

```python
c = torch.tensor([[77, 114, 140, 191, 81],
                  [39, 56, 46, 119, 17],
                  [61, 29, 20, 33, 160]])
print('c = ', c)
```

```plaintext
c =  tensor([[ 77, 114, 140, 191],
        [ 39,  56,  46, 119],
        [ 61,  29,  20,  33]])
```

```python
import matplotlib.pyplot as plt

plt.axis('off') # or plt.xticks([]); plt.yticks([])
plt.imshow(c, cmap='gray', vmin=0, vmax=255)
plt.show()
```

![[Pasted image 20240805141846.png]]


#### 3-D Tensor

- 언어적 표현: _동일한 크기의_ 2D tensor들이 여러 개 쌓인 입체적인 배열
- 대수적 표현: $2\times2\times3$ array 

![[Pasted image 20240805142936.png|500]]
- 공간 표현: 
	- 위 대수 표현은 각각 `(height, width) = (2,2)`인 <span style='color:red'>R</span>, <span style='color:green'>G</span>, <span style='color:blue'>B</span>로 표현됨:
	
	![[Pasted image 20240805150146.png|700]]
	- 이를 `torch.tensor`로 표현하면 위 대수적 표현과 같이 인식하고, 아래와 같은 그림으로 표현됨:
		
		![[Pasted image 20240805171551.png|300]]
- 코드 표현: 

```python
import pprint
# 채널별로 Tensor 생성
red_channel = torch.tensor([[255, 0],
                            [0, 255]])
green_channel = torch.tensor([[0, 255],
                              [0, 255]])
blue_channel = torch.tensor([[0, 0],
                             [255, 0]])

# 세 채널을 tuple로 결합해서 하나의 3차원 텐서로 합침
# dim은 확장하고자 하는 차원 입력: 세 번째 차원
d = torch.stack((red_channel, green_channel, blue_channel), dim=2)

print('d = ', d)
```

```
d =  tensor([[[255,   0,   0],
         [  0, 255,   0]],

        [[  0,   0, 255],
         [255, 255,   0]]])
```

```python
plt.axis('off') # plt.xticks([]), plt.yticks([])
plt.imshow(d)
plt.show()
```

![[Pasted image 20240805172050.png|300]]

#### N-D Tensor

- 언어적 표현 ($N\ge 4$): 동일한 크기가 $(N-1)$D tensor들이 여러 개 쌓인 입체적인 배열 구조
	- 예: 이미지에 시간 축/차원이 추가된 영상 = 4D tensor
- 공간 표현: 

![[Pasted image 20240805172757.png|500]]
![[Pasted image 20240805173235.png|500]]


## Reference

- Raschka, S., Liu, Y. & Mirjalili, V. (2022). Machine learning with PyTorch and scikit-learn. Birmingham: Packt.

---

# 2. Data Type & Basic Functions

## 2.1 Data Type

- **데이터 타입**이란 숫자형(`numeric`), 문자형(`string`) 등 데이터 값의 유형을 의미함
- `torch.tensor`에서 지원하는 데이터 유형은 크게 두 가지:
	- _정수형(integer)_, _실수형(float)_

### Integer Type

- 정수형 타입은 _소수가 없는 정수를 표현하는 데 쓰이는 데이터 타입_

#### `uint8`: 8bit _부호 없는(unsigned)_ 정수

- 언어적 표현: 8개 bit를 사용해 0~255(256개)의 정수를 표현
	- $[0, 255] = [0, 2^8-1] = \left[0,\sum_{k=0}^72^k\right]$

- 공간 표현: 

	![[Pasted image 20240805175827.png|700]]
- 코드 표현: **`dtype = torch.uint8`**

```python
a = torch.tensor(123, dtype = torch.uint8)
print('a = ', a)
print('a.dtype =', a.dtype)
```

```
a =  tensor(1, dtype=torch.uint8)
a.dtype = torch.uint8
```

```python
# torch.uint8의 최대값과 최소값 출력
max_value = torch.iinfo(torch.uint8).max
min_value = torch.iinfo(torch.uint8).min

print(f"torch.uint8 최대값: {max_value}")
print(f"torch.uint8 최소값: {min_value}")
```

```
torch.uint8 최대값: 255
torch.uint8 최소값: 0
```

- 음수를 사용하는 경우: 
	- _`uint8`, `uint16`, `uint32`은 음수가 사용되는데 `uint64`는 에러 남_.
		- 왜?

```python
c = torch.tensor([-1, -2, -3], dtype = torch.uint8)
print('c = ', c)
print('c.dtype =', c.dtype) # 이는 uint8에서 0~255의 마지막 값부터 역순으로 인식함
```

```
c =  tensor([255, 254, 253], dtype=torch.uint8)
c.dtype = torch.uint8
```
#### `int8`: 8bit _부호 있는_ 정수

- 언어적 표현: 8개 bit를 사용해 -128 ~ 127(256개)의 정수를 표현
	- $-2^6\sim2^6-1$
- 공간 표현: 

	![[Pasted image 20240805180258.png|700]]
- 코드 표현: **`dtype=torch.int8`**

```python
b = torch.tensor(-123, dtype = torch.int8)
print('b = ', b)
print('b.dtype =', b.dtype)
```

```
b =  tensor(-123, dtype=torch.int8)
b.dtype = torch.int8
```

```python
# torch.int8의 최대값과 최소값 출력
max_value = torch.iinfo(torch.int8).max
min_value = torch.iinfo(torch.int8).min

print(f"torch.int8 최대값: {max_value}")
print(f"torch.int8 최소값: {min_value}")
```

```
torch.int8 최대값: 127
torch.int8 최소값: -128
```
#### `int16`: 16bit 부호 있는 정수

- 언어적 표현: 16개 bit를 사용해 -32,768 ~ 32,767($2^{16}$개)의 정수 표현
- 코드 표현: **`dtype=torch.int16`** or **`dtype=torch.short`**

```python
d = torch.tensor(1, dtype = torch.short)
print('d = ', d)
print('d.dtype =', d.dtype)
```

```
d =  tensor(1, dtype=torch.int16)
d.dtype = torch.int16
```

```python
# torch.int16의 최대값과 최소값 출력
max_value = torch.iinfo(torch.int16).max
min_value = torch.iinfo(torch.int16).min

print(f"torch.int16 최대값: {format(max_value, ',d')}")
print(f"torch.int16 최소값: {format(min_value, ',d')}")
```

```
torch.int16 최대값: 32,767
torch.int16 최소값: -32,768
```

#### `int32`: 32bit 부호 있는 정수

- 언어적 표현: 32개 bit를 사용해 -2,147,483,648 ~ 2,147,483,647($2^{32}$개)의 정수 표현
	- 표준 정수 타입
- 코드 표현: **`dtype=torch.int32`** or **`torch.int`**

```python
e = torch.tensor(2, dtype = torch.int)
print('e = ', e)
print('e.dtype =', e.dtype)
```

```
e =  tensor(2, dtype=torch.int32)
e.dtype = torch.int32
```

```python
# torch.int32의 최대값과 최소값 출력
max_value = torch.iinfo(torch.int32).max
min_value = torch.iinfo(torch.int32).min

print(f"torch.int32 최대값: {format(max_value, ',d')}")
print(f"torch.int32 최소값: {format(min_value, ',d')}")
```

```
torch.int32 최대값: 2,147,483,647
torch.int32 최소값: -2,147,483,648
```

#### `int64`: 64bit 부호 있는 정수

- 언어적 표현: 64개 bit로 -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807 ($2^{64}$개)의 정수 표현
- 코드 표현: **`dtype=torch.int64`** or **`torch.long`**

```python
f = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype = torch.long) # or torch.int64
print('f = ', f)
print('f.dtype =', f.dtype)
```

```
f =  tensor([[1, 2, 3],
        [4, 5, 6]])
f.dtype = torch.int64
```

```python
# torch.int64의 최대값과 최소값 출력
max_value = torch.iinfo(torch.long).max
min_value = torch.iinfo(torch.long).min

print(f"torch.int64 최대값: {format(max_value, ',d')}")
print(f"torch.int64 최소값: {format(min_value, ',d')}")
```

```
torch.int64 최대값: 9,223,372,036,854,775,807
torch.int64 최소값: -9,223,372,036,854,775,808
```

### Floating Type

- 실수형 타입은 32bit 부동 소수점(floating point)와 64bit 부동 소수점 수 등이 있음
	- 신경망의 수치 계산에 주로 사용:
		- **왜?** 아마도, 최적화의 대상인 손실 함수가 실수형이어서 인듯

#### 고정 소수점 수(fixed point)

- 언어적 표현: 16bit로 숫자를 _정수부_와 _소수부_로 표현
- 공간 표현: 

![[Pasted image 20240805183222.png|700]]

- 고정 소수점의 문제: 102.005와 같은 수를 고정 소수점으로 표현하기 위해선 소수부의 각 자릿수를 따로 저장해야 함
	- 즉, 소수부 각 자리마다 4bit가 필요
	- 이는 컴퓨터 메모리의 낭비

#### 부동 소수점(floating point)

- 부동 소수점은 숫자를 _정규화(normalize)_하여 _지수부(exponent)_와 _가수부(mantissa)_로 나누어 표현
	- 아래와 같이 좌변의 원래 숫자를 우변처럼 `가수부 x 지수부` 꼴로 표현
	$$102.5 = 1.025\times 10^2$$
		- 이때 우변의 첫 번째 항을 가수부, 두 번째 항 $10$의 지수 부분을 지수부라고 함


#### `float16`: 16bit 부동 소수점 

- 언어적 표현: 16개 bit로 지수부와 가수부로 숫자를 표현
- 공간 표현: 

![[Pasted image 20240805184329.png|700]]

- 코드 표현: **`dtype=torch.float16`** or **`torch.half`**

```python
g = torch.tensor(1, dtype = torch.half)
print('g = ', g)
print('g.dtype =', g.dtype)
```

```
g =  tensor(1., dtype=torch.float16)
g.dtype = torch.float16
```

```python
# torch.float16의 최대값과 최소값 출력
max_value = torch.finfo(torch.half).max
min_value = torch.finfo(torch.half).min
absmin_value = torch.finfo(torch.half).smallest_normal

print(f"torch.float16 최대값: {max_value}")
print(f"torch.float16 최소값: {min_value}")
print(f"torch.float16 최소 절대값: {absmin_value}")
```

```
torch.float16 최대값: 65504.0
torch.float16 최소값: -65504.0
torch.float16 최소 절대값: 6.103515625e-05
```

#### `float32`: 32bit 부동 소수점

- 언어적 표현: 32개 bit로 숫자를 지수부와 가수부로 표현
- 공간 표현: 

	![[Pasted image 20240805184858.png|800]]
- 코드 표현: **`dtype=torch.float32`** or **`torch.float`**

```python
h = torch.tensor(1, dtype = torch.float)
print('h = ', h)
print('h.dtype =', h.dtype)
```

```
h =  tensor(1.)
h.dtype = torch.float32
```

```python
# torch.float32의 최대값과 최소값 출력
max_value = torch.finfo(torch.float32).max
min_value = torch.finfo(torch.float32).min
absmin_value = torch.finfo(torch.float32).smallest_normal

print(f"torch.float32 최대값: {max_value}")
print(f"torch.float32 최소값: {min_value}")
print(f"torch.float32 최소 절대값: {absmin_value}")
```

```
torch.float32 최대값: 3.4028234663852886e+38
torch.float32 최소값: -3.4028234663852886e+38
torch.float32 최소 절대값: 1.1754943508222875e-38
```

- `float32` 메모리 크기 확인해 보기:

```python
# .element_size(): 텐서 각 요소의 크기를 byte 단위로 반환
# .numel(): 텐사의 요소 수 반환
memory_size = h.element_size() * h.numel()

print(f"텐서의 메모리 크기: {memory_size} 바이트")
```

```
텐서의 메모리 크기: 4 바이트
```

#### `float64`: 64bit 부동 소수점

- 언어적 표현: 64개 bit로 숫자를 지수부와 가수부로 표현
- 코드 표현: **`dtype=torch.float64`** or **`torch.double`**

```python
i = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype = torch.double)
print('i = ', i)
print('i.dtype =', i.dtype)
```

```
i =  tensor([[1., 2., 3.],
        [4., 5., 6.]], dtype=torch.float64)
i.dtype = torch.float64
```

```python
# torch.double의 최대값과 최소값 출력
max_value = torch.finfo(torch.double).max
min_value = torch.finfo(torch.double).min
absmin_value = torch.finfo(torch.double).smallest_normal

print(f"torch.double 최대값: {max_value}")
print(f"torch.double 최소값: {min_value}")
print(f"torch.double 최소 절대값: {absmin_value}")
```

```
torch.double 최대값: 1.7976931348623157e+308
torch.double 최소값: -1.7976931348623157e+308
torch.double 최소 절대값: 2.2250738585072014e-308
```

```python
# .element_size(): 텐서 각 요소의 크기를 byte 단위로 반환
# .numel(): 텐사의 요소 수 반환
memory_size = i.element_size() * i.numel()

print(f"텐서의 메모리 크기: {memory_size} 바이트")
```

```
텐서의 메모리 크기: 48 바이트
```


이외에도 복소수를 표현하는 `torch.complex64`등과 논리형(boolean) 값을 나타내는 `torch.bool`이 있다.

### Type Casting

- 타입 캐스팅이란, 데이터의 자료형을 다르 데이터 타입으로 변경하는 것을 말함
	- type conversion, coercion이라고도 함
	- DL에서는 모델의 매개변수와 기울기(gradient)를 저장할 때 메모리에 부담을 줄이기 위해 사용됨

- 코드 표현: 
	- `i`를 `int8`의 tensor로 생성했을 때, `float32`과 `float64`로 각각 변경

```python
i = torch.tensor([2, 3, 4], dtype = torch.int8)
j = i.float()
k = i.double()

print('j.dtype = ', j.dtype)
print('k.dtype = ', k.dtype)
```

```
j.dtype =  torch.float32
k.dtype =  torch.float64
```

## 2.2 Tensor의 기초 함수 및 메서드


### 기초 함수

- 함수: tensor의 요소를 반환하거나, 요소를 이용해 계산을 수행

#### `min()`

- 언어적 표현: tensor의 모든 요소들 중 **최소값**을 반환
- 공간 표현: 

![[Pasted image 20240805204558.png|400]]
- 코드 표현: 

```python
l = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]], dtype = torch.float)

print('torch.min(l) =', torch.min(l))
```

```
torch.min(l) = tensor(1.)
```

#### `max()`

- 언어적 표현: tesor의 모든 요소들 중 **최대값**을 반환
- 공간 표현: 

![[Pasted image 20240805204821.png|400]]
- 코드 표현: 

```python
print('torch.max(l) =', torch.max(l))
```

```
torch.max(l) = tensor(8.)
```

#### `sum()`, `prod()`, `mean()`, `var()`, `std()`

- **`sum()`**: tensor의 모든 요소들의 **합**을 계산
- **`prod()`**: tensor의 모든 요소들의 **곱**을 계산
- **`mean()`**: tensor의 요소들의 **평균**을 계산
- **`var()`**: tensor의 요소들의 **표본분산**을 계산
- **`std()`**: tensor의 요소들의 **표본표준편차**를 계산

- 코드 표현: 

```python
print('torch.sum(l) =', torch.sum(l))
print('torch.prod(l) =', torch.prod(l))
print('torch.mean(l) =', torch.mean(l))
print('torch.var(l) =', torch.var(l))
print('torch.std(l) =', torch.std(l))
```

```
torch.sum(l) = tensor(36.)
torch.prod(l) = tensor(40320.)
torch.mean(l) = tensor(4.5000)
torch.var(l) = tensor(6.)
torch.std(l) = tensor(2.4495)
```

> [!warning] 
> - `torch.tensor()`의 `dtype` 인자의 default 값은 `int32`인데, `mean()`, `var()`, `std()`는 `dtype`이 `float` 형이어야 한다. `int`형이면 에러가 난다. 

### Tensor의 특성을 확인하는 메서드

- **`.dim()`**: 차원 개수 반환
- **`.size()`**: tensor의 크기/모양 반환
	- **`.shape`** 속성(attribute)도 동일한 기능
- **`.numel()`**: tensor가 가지고 있는 요소 개수 반환

```python
print('l.dim() = ', l.dim())
print('l.size() = ', l.size())
print('l.shape = ', l.shape)
print('l.numel() = ', l.numel())
```

```
l.dim() =  2
l.size() =  torch.Size([2, 4])
l.shape =  torch.Size([2, 4])
l.numel() =  8
```

---

