import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=720, freq='D')
data = np.random.randn(720) * 10 + 50

# 2020,2021년 한국 공휴일 Dataframe 생성
holidays = make_holidays_df(year_list=[2020,2021], country='KR')

df = pd.DataFrame({
    'ds' : dates,
    'y' : data
})

# Model에 holiday 적용
model = Prophet(holidays=holidays)
model.fit(df)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

model.plot(forecast)
plt.savefig('./prophet/holiday1.png')

model.plot_components(forecast)
plt.savefig('./prophet/holiday2.png')