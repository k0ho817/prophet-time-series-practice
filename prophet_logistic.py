import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = np.random.randn(100) * 30 + 50
# 0 ~ 100 사이의 값으로 클리핑
data = np.clip(data, 0, 100)

df = pd.DataFrame({
    'ds' : dates,
    'y' : data
})

model = Prophet(growth='logistic')
# logistic 함수를 이용할때에는 최대값 최소값을 데이터 프레임에 추가해 줘야 함
df['cap'] = 100
df['floor'] = 0
model.fit(df)

future = model.make_future_dataframe(periods=30)
future['cap'] = 100
future['floor'] = 0

forecast = model.predict(future)

model.plot(forecast)
plt.savefig('./prophet/logi1.png')

model.plot_components(forecast)
plt.savefig('./prophet/logi2.png')