import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=720, freq='D')
data = np.random.randn(720) * 10 + 50

df = pd.DataFrame({
    'ds' : dates,
    'y' : data
})

# Model에 yearly_seasonality 적용
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

# Model에 monthly 라는 이름의 달 단위 푸리에 계수 8로 지정하여 seasonality 적용
# 푸리에 계수가 높을수록 더 세밀하고 복잡한 곡선 생성 가능
# 하지만 너무 높게 설정할 경우 노이즈 까지 학습하여 예측력이 저하될 가능성 존재
model.add_seasonality(name='monthly', period=30.5, fourier_order=8)

model.fit(df)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

model.plot(forecast)
plt.savefig("./prophet/season1.png")

model.plot_components(forecast)
plt.savefig("./prophet/season2.png")