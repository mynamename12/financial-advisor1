
from prophet import Prophet
import pandas as pd

df = pd.read_csv("synthetic_financial_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df[df["type"] == "مصروف"]
df = df.groupby("date")["amount"].sum().reset_index()
df.columns = ["ds", "y"]

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

forecast[["ds", "yhat"]].to_csv("forecast_result.csv", index=False)
