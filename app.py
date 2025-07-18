
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# تحميل النموذج
with open("hybrid_financial_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = pd.read_csv(file)

    df["description"] = df["description"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.weekday

    label_encoder = LabelEncoder()
    df["type_encoded"] = label_encoder.fit_transform(df["type"])
    df["clean_description"] = df["description"].str.lower()

    predictions = model.predict(df[["clean_description", "amount", "day_of_month", "day_of_week", "type_encoded"]])
    df["category"] = predictions

    total_income = df[df["type"] == "دخل"]["amount"].sum()
    total_expense = df[df["type"] == "مصروف"]["amount"].sum()
    risk_score = round((total_expense / total_income) * 100, 2) if total_income > 0 else 100

    # تحذير خطر الإفلاس
    balance = 10000
    net_loss = total_expense - total_income
    if net_loss > 0:
        daily_deficit = net_loss / len(df["date"].unique())
        days_to_zero = int(balance / daily_deficit) if daily_deficit > 0 else 0
        bankruptcy_risk = f"⚠️ وفق نمطك المالي، قد تنفد أموالك خلال {days_to_zero} يوم."
    else:
        bankruptcy_risk = "✅ لا يوجد خطر إفلاس حالياً. مصروفك أقل من دخلك."

    top_categories = df["category"].value_counts().head(5).to_dict()

    advice = []
    if risk_score > 80:
        advice.append("⚠️ مصروفاتك تقترب من دخلك، قلل النفقات فورًا.")
    if "مطاعم" in top_categories:
        advice.append("📉 حاول تقليل الإنفاق على المطاعم.")

    return jsonify({
        "total_income": total_income,
        "total_expense": total_expense,
        "risk_score": risk_score,
        "top_categories": top_categories,
        "bankruptcy_risk": bankruptcy_risk,
        "advice": advice
    })

if __name__ == '__main__':
    app.run(debug=True)
