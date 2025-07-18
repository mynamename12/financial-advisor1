import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# تحميل البيانات
df = pd.read_csv("synthetic_financial_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df[df["type"] == "مصروف"]

# تجميع المصروفات يوميًا
df = df.groupby("date")["amount"].sum().reset_index()
df = df.set_index("date")

# بناء النموذج باستخدام Holt-Winters
model = ExponentialSmoothing(df["amount"], trend="add", seasonal="add", seasonal_periods=7)
model_fit = model.fit()

# التنبؤ لـ 30 يوم قادمة
forecast = model_fit.forecast(30)

# تحويل التنبؤ إلى DataFrame
forecast_df = forecast.reset_index()
forecast_df.columns = ["ds", "yhat"]
forecast_df.to_csv("forecast_result.csv", index=False)
