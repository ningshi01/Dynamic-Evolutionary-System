import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv('test_data.csv')
# 确保 ds 列是 datetime 类型
df['ds'] = pd.to_datetime(df['ds'])

# 2. 创建并拟合模型
model = Prophet() # 创建模型实例
model.fit(df)     # 拟合数据

# 3. 构建未来时间数据框
future = model.make_future_dataframe(periods=60, freq='min', include_history=True) # 未来60min的预测
# 查看 future 的尾部，确认包含了未来的日期
print(future)

# 4. 进行预测
forecast = model.predict(future)
# 查看预测结果的列：ds, yhat, yhat_lower, yhat_upper 等]
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# 5. 绘制预测结果
fig1 = model.plot(forecast)
plt.title('Sales Forecast')
plt.show()

# 6. 绘制组件（核心优势！）
fig2 = model.plot_components(forecast)
plt.show()

# 保存
forecast.to_csv('create_data_v0.csv', index=False)