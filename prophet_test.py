import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = np.random.randn(100) * 10 + 50

df = pd.DataFrame({
    'ds' : dates,
    'y' : data
})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

model.plot(forecast)
plt.savefig('./prophet/test1.png')

model.plot_components(forecast)
plt.savefig('./prophet/test2.png')