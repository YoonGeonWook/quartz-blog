---
sticker: emoji//1f4d8
tags:
  - Deep_Learning
  - AI_Tech
  - BoostCamp
  - Pytorch
  - tensor
  - Dataset
  - DataLoader
  - Classification
  - PalmerPenguins
---
# 9~10. Binary Classification

두 챕터에 대해서도 어느정도 알고 있다고 생각하여 추가적으로 숙지하고 싶으면 좋겠다 싶은 내용 위주로 정리하고, 이번에도 다른 데이터셋으로 분류 문제를 풀어본다. 

#### `torch.utils.data.DataLoader()`

- 참고: 
	- https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
	- https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html
	- https://velog.io/@tjdtnsu/PyTorch-%EA%B8%B0%EC%B4%88-DataLoader-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0


데이터 로더(Data loader)는 데이터셋과 샘플러(sampler)를 결합해서 주어진 데이터셋을 순회하여 batch samples를 반환한다.

일반적으로 주어진 `Dataset`으로부터 특정 크기의 미니배치(mini-batch)로 가져와서 매 에폭(epoch)마다 데이터를 섞어서 과적합(overfitting)을 방지하고, `multiprocessing`을 통해 데이터 검색 속도를 높이고자 하는데, 이를 간단한 API로 추상화한 iterable 객체가 `DataLoader`인 것이다. 

- **`torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, ...)`**

보통은 위와 같이 3개의 인자를 주로 사용하는데, 가끔씩 `num_workers`와 `pin_memory` 인자도 고려한다.

- **`num_workers`**: 데이터 로딩을 위한 병렬 작업자(worker) 프로세스 수를 설정하는 것
	- default: `0`
	- 효과: 
		- 병렬 처리: `num_workers`를 `0`보다 큰 값으로 설정하면 데이터 로드를 여러 프로세스로 분산하여 병렬로 수행
		- CPU가 데이터를 빠르게 로딩해서 GPU 연산 시간의 비율을 높이기 위한 작업이라고 할 수 있음
		- 너무 많은 작업자를 설정하면 오버헤드가 걸릴 수 있음
		- 보통 작은 데이터셋에서 더 선호됨
- **`pin_memory`**: `True`로 설정하면 `DataLoader`가 데이터를 로드할 때 CUDA 메모리로 직접 복사(pinned memory)하여 GPU로의 전송 속도를 높인다. 
	- default: `False`
	- 효과: 
		- 고속 메모리 전송: 고정된(pinned) 메모리를 사용해 CPU에서 GPU로 데이터를 전송하는 속도가 빨라짐
			- 비교적 큰 데이터셋을 처리할 때 효과가 있음
		- GPU 사용할 때만 유효함

#### BCE Loss, BCE With Logit Loss, Cross-Entropy Loss

- 참고: 
	- https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
	- https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
	- https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

수업 시간에는 Binary Cross-Entropy(BCE) Loss에 대해서만 다루었다. PyTorch 공식 문서를 참고해서 `BCELoss`, `BCEWithLogitLoss`, `CrossEntropyLoss`에 대해 살펴본다. 

- **`nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')`**
	- 예측 확률 $y$과 target $t$가 주어질 때 이진 교차 엔트로피 손실을 반환
	- Parameters: 
		- **`weight`**: (Tensor, optional)
			- 각 (배치) 샘플에 부여하는 가중치 텐서: 손실 함수 값을 가중 평균/합으로 계산할 때 사용
			- 기본값: `None` - 모든 샘플에 동일한 가중치
		- `size_average`: (bool, optional)
			- 손실 값을 평균화할지 여부 결정
			- 기본값: `True` - PyTorch 0.4.0 이후에는 사용이 권장되지 않고, 대신 `reduction` 인자를 사용함
		- `reduce`: (bool, optional)
			- `True`로 설정하면 전체 mini-batch에 대해 손실을 계산하고, `False`로 설정하면 개별 샘플의 손실을 계산
			- 기본값: `True` - PyTorch 0.4.0 이후에는 사용이 권장되지 않고, 대신 `reduction` 인자를 사용함
		- **`reduction`**: (str, optional)
			- `'none'`, `'mean'`, `'sum'` 중 하나 선택
				- `'none'`: 개별 샘플의 손실을 반환
					- batch size = 32인 경우 해당 배치 데이터에 대해 32개의 손실 값을 반환
				- `'mean'`(default): 배치 데이터의 평균 손실값을 반환
				- `'sum'`: 배치 데이터의 손실값 합계를 반환

예측 클래스 확률 $y$와 target $t$가 주어졌을 때, `reduction = 'none'`인 경우의 손실함수를 살펴보자. 

$$
\ell(y,t)=L=\{l_1,\ldots,l_N\}^T,\quad l_n=-w_n(t_n\log y_n+(1-t_n)\log(1-y_n))
$$
여기서 $N$은 배치 크기를 나타낸다. `reduction = 'none'`이 아니면 다음과 같이 사용된다.

$$
\ell(y,t) = \cases{\text{mean(L)},\quad \text{if reduction = 'mean'}\\\text{sum}(L),\quad \text{if reduction = 'sum'}}
$$

- **`BCEWithLogitLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)`**: 
	- 이는 이진 분류 문제에서 자주 사용되는 손실 함수로, Sigmoid 함수와 BCE 손실을 결합하여 더욱 안정적인 계산을 제공한다. 
	- Parameters: 
		- **`weight`**: (Tensor, optional)
			- 손실 함수 각 요소에 대한 가중치
			- 기본값: `None`
		- **`reduction`**: (str, optional)
			- `'none'`, `'mean'`, `'sum'` 중 하나
			- 기본값: `'mean'`
		- **`pos_weight`**: (Tensor, optional)
			- 클래스 불균형이 있을 경우, 양성(positive) 클래스에 부여할 가중치. 드물게 발생하는 클래스에 더 높은 가중치를 부여할 수 있음

`nn.BCEWithLogitsLoss`는 Sigmoid 함수와 BCE 손실을 결합한 손실함수다. 이렇게 결합함으로써 개별적으로 Sigmoid를 사용하고 그 뒤에 BCE 손실을 사용하는 것보다 수치적으로 더 안정적인 계산이 가능하고, 이를 위해 log-sum-exp 트릭을 사용해 수치적 안정성을 보장한다.

- `reduction = 'none'`인 경우 손실함수는 다음과 같이 정의된다:
	- 여기서 $N$은 배치 크기

$$
\ell(y,t)=L=\{l_1,\ldots,l_N\}^T,\quad l_n=-w_n(t_n\log\sigma(y_n)+(1-t_n)\log(1-\sigma(y_n)))
$$

- `reduction = 'mean'`이면 손실의 평균을, `'sum'`이면 손실의 합을 반환한다:

$$
\ell(y,t) = \cases{\text{mean(L)},\quad \text{if reduction = 'mean'}\\\text{sum}(L),\quad \text{if reduction = 'sum'}}
$$

- 다중 라벨 분류(multi-label classification)의 경우 손실 함수는 다음과 같이 정의된다. 
	- 여기서 $c$는 $c$번째 클래스를 의미한다.

$$
\ell_c(y,t) = L_c=\{l_{1c},\ldots,l_{Nc}\}^T,\quad l_{nc} = -w_{nc}\left(p_ct_{nc}\log\sigma(y_{nc})+(1-t_{nc})\log(1-\sigma(y_{nc}))\right)
$$

- $p_c>1$이면 recall이 증가하고, $p_c<1$이면 precision이 증가한다.
	- 예: 단일 클래스의 양성 샘플이 100개이고 음성 샘플이 300개인 경우 `pos_weight = 3`이면 이는 데이터셋에 양성 샘플이 300개인 것처럼 동작한다.

- **`torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)`**
	- 다중 클래스 분류 문제를 위한 손실 함수
	- 이는 Softmax 활성화와 음의 로그 가능도(Negative Log-likelihood) 손실을 결합한 것
	- Parameter: 
		- **`weight`**: (Tensor, optional) 
			- 각 클래스에 대한 가중치를 지정하는 1D 텐서
			- 클래스 불균형을 처리
		- **`ignore_index`**: (int, optional)
			- 손실 계산에서 무시할 target 값을 지정
			- 패딩 토큰을 무시할 때 유용
		- **`label_smoothing`**: (float `[0,1]`, optional)
			- 레이블을 원-핫 인코딩에서 약간 벗어나게 하여 모델이 극단적인 확률 값을 예측하지 않도록 하여 과적합을 방지할 수 있음
			- 실제 레이블 대신에 `(1-label_smoothing)` 확률을 클래스에 할당하고 나머지 확률을 다른 클래스에 분배한다.

`torch.nn.CrossEntropy()`는 입력된 로짓(logit)과 target 사이의 교차 엔트로피 손실을 계산한다. 분류 문제에 유용하며, 클래스 불균형이 있는 경우 `weight` 인자를 사용하여 가중치를 할당할 수 있다.

입력은 각 클래스에 대한 정규화되지 않은 로짓(logit)이어야 하며, 크기는 `(minibatch, C)` 또는 `(minibatch, C, d1, d2, ..., dK)` 형식을 갖는다 . `K`는 1 이상이어야 하며, 고차원 입력에 유용하다. 

## Multi-class Classification

#### Palmer Penguins Data

- https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data


![[Pasted image 20240808203712.png|500]]출처: [Artwork by @allison_horst](https://allisonhorst.github.io/palmerpenguins/)

- 총 344개 행과 7개의 변수로 이루어진 데이터
- 변수: 
	- `species`: 펭귄 종류 - Adelie, Chinstrap, Gentoo
	- `island`: 펭귄 서식지 - Torgersen, Biscoe, Dream
	- `sex`: 펭귄 성별 - MALE, FEMALE
	- `culmen_length_mm`: 부리 길이
	- `culmen_depth_mm`: 부리 깊이
	- `flipper_length_mm`: 물갈퀴 길이
	- `body_mass_g`: 펭귄 몸무게

```python
# Donwload dataset from kaggle
!kaggle datasets download -d parulpandey/palmer-archipelago-antarctica-penguin-data
# unzip zip file
!unzip palmer-archipelago-antarctica-penguin-data.zip
```

```python
import pandas as pd
import numpy as np

data = pd.read_csv("penguins_size.csv", sep=',', header=0)
data.head()
```

| index | species | island    | culmen\_length\_mm | culmen\_depth\_mm | flipper\_length\_mm | body\_mass\_g | sex    |
| ----- | ------- | --------- | ------------------ | ----------------- | ------------------- | ------------- | ------ |
| 0     | Adelie  | Torgersen | 39\.1              | 18\.7             | 181\.0              | 3750\.0       | MALE   |
| 1     | Adelie  | Torgersen | 39\.5              | 17\.4             | 186\.0              | 3800\.0       | FEMALE |
| 2     | Adelie  | Torgersen | 40\.3              | 18\.0             | 195\.0              | 3250\.0       | FEMALE |
| 3     | Adelie  | Torgersen | NaN                | NaN               | NaN                 | NaN           | NaN    |
| 4     | Adelie  | Torgersen | 36\.7              | 19\.3             | 193\.0              | 3450\.0       | FEMALE |
```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 344 entries, 0 to 343
Data columns (total 7 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   species            344 non-null    object 
 1   island             344 non-null    object 
 2   culmen_length_mm   342 non-null    float64
 3   culmen_depth_mm    342 non-null    float64
 4   flipper_length_mm  342 non-null    float64
 5   body_mass_g        342 non-null    float64
 6   sex                334 non-null    object 
dtypes: float64(4), object(3)
memory usage: 18.9+ KB
```

#### 결측치 최빈값으로 대체

- 각 열의 최빈값으로 결측치 대체

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
data.iloc[:,:] = imputer.fit_transform(data)
```

```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 344 entries, 0 to 343
Data columns (total 7 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   species            344 non-null    object 
 1   island             344 non-null    object 
 2   culmen_length_mm   344 non-null    float64
 3   culmen_depth_mm    344 non-null    float64
 4   flipper_length_mm  344 non-null    float64
 5   body_mass_g        344 non-null    float64
 6   sex                344 non-null    object 
dtypes: float64(4), object(3)
memory usage: 18.9+ KB
```

#### 범주형 데이터 처리

```python
print(data.sex.value_counts())
```

```
sex
MALE      178
FEMALE    165
.           1
Name: count, dtype: int64
```

```python
data = data[data.sex!='.']
```

- `island`, `sex`: One-Hot Encoder로 처리

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
data = pd.concat([
    data[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']],
    pd.DataFrame(ohe.fit_transform(data[['island', 'sex']]).toarray())
], axis=1).reset_index()
```

- Target인 `species` 이산화

```python
data['species'] = data['species'].map({'Adelie':0, 'Gentoo':1, 'Chinstrap':2})
data.head()
```

| level\_0 | index | species | culmen\_length\_mm | culmen\_depth\_mm | flipper\_length\_mm | body\_mass\_g | 0    | 1    | 2    | 3    | 4    |
| -------- | ----- | ------- | ------------------ | ----------------- | ------------------- | ------------- | ---- | ---- | ---- | ---- | ---- |
| 0        | 0     | 0       | 39\.1              | 18\.7             | 181\.0              | 3750\.0       | 0\.0 | 0\.0 | 1\.0 | 0\.0 | 1\.0 |
| 1        | 1     | 0       | 39\.5              | 17\.4             | 186\.0              | 3800\.0       | 0\.0 | 0\.0 | 1\.0 | 1\.0 | 0\.0 |
| 2        | 2     | 0       | 40\.3              | 18\.0             | 195\.0              | 3250\.0       | 0\.0 | 0\.0 | 1\.0 | 1\.0 | 0\.0 |
| 3        | 3     | 0       | 41\.1              | 17\.0             | 190\.0              | 3800\.0       | 0\.0 | 0\.0 | 1\.0 | 0\.0 | 1\.0 |
| 4        | 4     | 0       | 36\.7              | 19\.3             | 193\.0              | 3450\.0       | 0\.0 | 0\.0 | 1\.0 | 1\.0 | 0\.0 |

#### 데이터 분할 및 표준화 

- test size는 20%로 설정

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop(['index', 'species'], axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']] = scaler.fit_transform(X_train[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']])
X_test[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']] = scaler.transform(X_test[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']])
```

#### 텐서 생성

```python
import torch

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values)
```

#### `Dataset` & `DataLoader` 정의

```python
from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_dl = DataLoader(train_data, batch_size=16, shuffle=True)
test_dl = DataLoader(test_data, batch_size=16, shuffle=False)
```

- Code Check: 

```python
batched_X, batched_y = next(iter(train_dl))
assert batched_X.shape == (16, 9)
assert batched_y.shape == (16, )

batched_X, batched_y = next(iter(test_dl))
assert batched_X.shape == (16, 9)
assert batched_y.shape == (16, )

print("Shape of batch is correct!")
```

```
Shape of batch is correct!
```

#### Multi-layer Perceptron

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
import torch.nn as nn
class MLPModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)
```

```python
in_features = X_train_tensor.shape[1]
hidden_features, out_features = 100, 3
model = MultiClassModel(in_features, hidden_features, out_features)
model.to(device)
```

```
MLPModel(
  (linear1): Linear(in_features=9, out_features=100, bias=True)
  (relu): ReLU()
  (linear2): Linear(in_features=100, out_features=3, bias=True)
)
```

#### 손실 함수 & 옵티마이저

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
```

#### 모델 학습 및 평가 함수

```python
import matplotlib.pyplot as plt

def train(model, criterion, optimizer, dataloader, device, num_epochs=30):
    model.train()
    model.to(device)

    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        n_data = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == targets).sum().item()
            n_data += targets.size(0)
        train_acc = correct / n_data
        if epoch == 0 or (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}, Train Acc: {train_acc}")
        loss_list.append(epoch_loss/len(dataloader))
        acc_list.append(train_acc)
    plt.plot(range(1,epoch+2), loss_list, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1,epoch+2), acc_list, label="Training Acc.")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
```

```python
def test(model, criterion, dataloader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        running_loss = 0
        correct = 0
        n_data = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            loss = criterion(pred, targets)
            running_loss += loss.item()

            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == targets).sum().item()
            n_data += targets.size(0)
        acc = correct/n_data
    print(f'Loss: {running_loss/len(dataloader)}, Test Acc: {correct/n_data}')
```

```python
train(model, criterion, optimizer, train_dl, device, 100)
```



```
Epoch 1, Loss: 1.0479043821493785, Train Acc: 0.5036496350364964
Epoch 10, Loss: 0.8127342893017663, Train Acc: 0.9416058394160584
Epoch 20, Loss: 0.6732672287358178, Train Acc: 0.8321167883211679
Epoch 30, Loss: 0.5657252851459715, Train Acc: 0.8284671532846716
Epoch 40, Loss: 0.4902496453788545, Train Acc: 0.8394160583941606
Epoch 50, Loss: 0.44713707433806527, Train Acc: 0.8795620437956204
Epoch 60, Loss: 0.4133869624800152, Train Acc: 0.9051094890510949
Epoch 70, Loss: 0.37684515946441227, Train Acc: 0.927007299270073
Epoch 80, Loss: 0.36036207692490685, Train Acc: 0.9416058394160584
Epoch 90, Loss: 0.31337327924039626, Train Acc: 0.9598540145985401
Epoch 100, Loss: 0.281636506319046, Train Acc: 0.9671532846715328
```

![[Pasted image 20240809153234.png]]

```python
test(model, criterion, test_dl, device)
```

```
Loss: 0.2800576210021973, Test Acc: 0.9855072463768116
```

