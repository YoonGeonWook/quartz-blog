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
  - Linear-regression
  - Diamond
---
# 7~8. Linear Regression

- 선형 회귀: 
	- 학습 데이터를 사용해 feature와 target 사이의 선형 관계를 분석하고 이를 모델로 학습하여, unseen data를 연속적인 숫자 값으로 예측하는 과정
- 클래스란?
	- _인스턴스(객체)를 생성하기 위한 틀_
		- 메서드와 속성을 정의함
- 인스턴스란?
	- _클래스에서 생성된 구체적인 객체_

> [!note] 
> 수업에서 다루는 개념은 익숙한 것이 많아서 다른 데이터를 가지고 실습해서 정리했음


#### Diamond Data

- 총 53,940개 행과 10개의 변수로 이루어진 데이터
- 변수: 
	- `carat`: 다이아몬드 무게
	- `cut`: 다이아몬드 컷팅의 품질
	- `color`: 보석 색상 품질
	- `clarity`: 투명도
	- `x`: 길이
	- `y`: 너비
	- `z`: 깊이
	- `table`: 다이아몬드 상단 너비
	- `price`: 가격
	- `depth`: total depth = `z / avg(x,y)`

```python
# Donwload dataset from kaggle
!kaggle datasets download -d shivam2503/diamonds
# unzip zip file
!unzip diamonds.zip
```

```python
import pandas as pd
import numpy as np

data = pd.read_csv("diamonds.csv", sep=',', header=0)
data.head()
```

| index | Unnamed: 0 | carat | cut     | color | clarity | depth | table | price | x     | y     | z     |
| ----- | ---------- | ----- | ------- | ----- | ------- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0     | 1          | 0\.23 | Ideal   | E     | SI2     | 61\.5 | 55\.0 | 326   | 3\.95 | 3\.98 | 2\.43 |
| 1     | 2          | 0\.21 | Premium | E     | SI1     | 59\.8 | 61\.0 | 326   | 3\.89 | 3\.84 | 2\.31 |
| 2     | 3          | 0\.23 | Good    | E     | VS1     | 56\.9 | 65\.0 | 327   | 4\.05 | 4\.07 | 2\.31 |
| 3     | 4          | 0\.29 | Premium | I     | VS2     | 62\.4 | 58\.0 | 334   | 4\.2  | 4\.23 | 2\.63 |
| 4     | 5          | 0\.31 | Good    | J     | SI2     | 63\.3 | 58\.0 | 335   | 4\.34 | 4\.35 | 2\.75 |

```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 53940 entries, 0 to 53939
Data columns (total 11 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  53940 non-null  int64  
 1   carat       53940 non-null  float64
 2   cut         53940 non-null  object 
 3   color       53940 non-null  object 
 4   clarity     53940 non-null  object 
 5   depth       53940 non-null  float64
 6   table       53940 non-null  float64
 7   price       53940 non-null  int64  
 8   x           53940 non-null  float64
 9   y           53940 non-null  float64
 10  z           53940 non-null  float64
dtypes: float64(6), int64(2), object(3)
memory usage: 4.5+ MB
```

```python
data = data.drop(["Unnamed: 0"], axis=1)
```

#### 범주형 변수 인코딩: `LabelEncoder()`

- 문자형 데이터를 알파벳 순서에 따라 고유한(unique)한 정수로 매핑하는 기법

```python
print("Unique values of categorical columns:\n")
print("cut:", data['cut'].unique())
print("color:", data['color'].unique())
print("clarity:", data['clarity'].unique())
```

```
Unique values of categorical columns:

cut: ['Ideal' 'Premium' 'Good' 'Very Good' 'Fair']
color: ['E' 'I' 'J' 'H' 'F' 'G' 'D']
clarity: ['SI2' 'SI1' 'VS1' 'VS2' 'VVS2' 'VVS1' 'I1' 'IF']
```

```python
from sklearn.preprocessing import LabelEncoder

# LabelEncoder 객체 생성
encoder = LabelEncoder()

# fit & transform
data['cut'] = encoder.fit_transform(data['cut'])
data['color'] = encoder.fit_transform(data['color'])
data['clarity'] = encoder.fit_transform(data['clarity'])

data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 53940 entries, 0 to 53939
Data columns (total 10 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   carat    53940 non-null  float64
 1   cut      53940 non-null  int64  
 2   color    53940 non-null  int64  
 3   clarity  53940 non-null  int64  
 4   depth    53940 non-null  float64
 5   table    53940 non-null  float64
 6   price    53940 non-null  int64  
 7   x        53940 non-null  float64
 8   y        53940 non-null  float64
 9   z        53940 non-null  float64
dtypes: float64(6), int64(4)
memory usage: 4.1 MB
```

```python
X = data.loc[:, ~data.columns.isin(["price"])]
y = data['price']
```

#### 데이터 분할

- test set의 크기를 전체 데이터셋의 20%로 설정

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
```
#### 수치형 변수 표준화

- 수치형 feature `carat`, `depth`, `table`, `x`, `y`, `z`를 표준화 

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_vars = ['carat', 'depth', 'table', 'x', 'y', 'z']

X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])
```

```python
import torch

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).unsqueeze(1)
```

#### Dataset & DataLoader 설정

```python
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
test_dl = DataLoader(test_data, batch_size=64, shuffle=False)
```

- Code Check: 

```python
batched_X, batched_y = next(iter(train_dl))
assert batched_X.shape == (64, 9)
assert batched_y.shape == (64, 1)

batched_X, batched_y = next(iter(test_dl))
assert batched_X.shape == (64, 9)
assert batched_y.shape == (64, 1)

print("Shape of batch is correct!")
```

```
Shape of batch is correct!
```

## 모델 학습

#### 모델 정의

```python
import torch.nn as nn

class LinearRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionNN, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1)
    
    def forward(self, x):
        return self.linear(x)
```

```python
input_size = X_train_tensor.shape[1]
model = LinearRegressionNN(input_size)
model.to(device)
```

#### 손실 함수 및 옵티마이저

```python
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
```

#### 모델 학습 및 평가 함수

```python
def train(model, criterion, optimizer, dataloader, device, num_epochs=100):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_data = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)

            optimizer.zero_grad()
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(targets)
            n_data += len(targets)
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1}, Loss(RMSE): {np.sqrt(epoch_loss/n_data)}")
```

```python
train(model, criterion, optimizer, train_dl, device, 50)
```

```
Epoch 10, Loss(RMSE): 1508.6326081292011
Epoch 20, Loss(RMSE): 1408.5340922479804
Epoch 30, Loss(RMSE): 1378.2940343497871
Epoch 40, Loss(RMSE): 1367.2134178554015
Epoch 50, Loss(RMSE): 1362.014450932358
```

```python
def test(model, dataloader, criterion, device):
    model.eval()
    model.to(device)

    test_loss = 0.0
    n_data = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            loss = criterion(pred, targets)
            test_loss += loss.item() * len(targets)
            n_data += len(targets)
    print(f"Test Loss(RMSE): {np.sqrt(test_loss/n_data)}")
```

```python
test(model, test_dl, criterion, device)
```

```
Test Loss(RMSE): 1357.3684051504604
```

## Scikit-Learn `LinearRegression()`과 비교

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f"Train Loss(RMSE): {rmse}")

y_pred = lm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test Loss(RMSE): {rmse}")
```

```
Train Loss(RMSE): 1352.8001458294523
Test Loss(RMSE): 1351.263479683125
```

---
