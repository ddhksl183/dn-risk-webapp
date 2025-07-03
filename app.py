from flask import Flask, request, render_template, send_file, redirect, url_for
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from io import BytesIO
import shap
import json

app = Flask(__name__)

# MySQL connection config
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Blackninja183!@localhost/diabetic_nephropathy'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model
class PredictionLog(db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    Diabetes_Status = db.Column(db.Integer)
    A1c_Percent = db.Column(db.Float)
    eGFR = db.Column(db.Float)
    Urine_Microalbumin_mg_L = db.Column(db.Float)
    Serum_Creatinine_umol_L = db.Column(db.Float)
    Fasting_Venous_Cholesterol_mmol_L = db.Column(db.Float)
    Age = db.Column(db.Integer)
    Ave_SBP_mmHg = db.Column(db.Float)
    BMI_kg_m2 = db.Column(db.Float)
    LDL_mmol_L = db.Column(db.Float)
    Risk_Score = db.Column(db.Float)
    Risk_Category = db.Column(db.String(20))
    Prediction_Time = db.Column(db.DateTime, default=datetime.utcnow)

# Load model and SHAP explainer
model = joblib.load("random_forest_top10.pkl")
explainer = shap.TreeExplainer(model)

# Risk categorization function
def assign_risk_category(risk_score):
    if risk_score < 0.475:
        return "Low Risk"
    elif risk_score < 0.875:
        return "Intermediate Risk"
    else:
        return "High Risk"

# Routes
@app.route("/")
def landing_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            input_data = [
               int(request.form["Diabetes_Status"]),
                float(request.form["A1c_Percent"]),
                float(request.form["eGFR"]),
                float(request.form["Urine_Microalbumin_mg_L"]),
                float(request.form["Serum_Creatinine_umol_L"]),
                float(request.form["Fasting_Venous_Cholesterol_mmol_L"]),
                int(request.form["Age"]),
                float(request.form["Ave_SBP_mmHg"]),
                float(request.form["BMI_kg_m2"]),
                float(request.form["LDL_mmol_L"])
            ]

            input_array = np.array(input_data).reshape(1, -1)
            risk_score = model.predict_proba(input_array)[0][1]
            risk_category = assign_risk_category(risk_score)

            # SHAP values
            shap_values = explainer.shap_values(input_array)
            feature_names = [
                "Diabetes_Status", "A1c_Percent", "eGFR", "Urine_Microalbumin_mg_L",
                "Serum_Creatinine_umol_L", "Fasting_Venous_Cholesterol_mmol_L",
                "Age", "Ave_SBP_mmHg", "BMI_kg_m2", "LDL_mmol_L"
            ]

            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_dict = dict(zip(feature_names, shap_values[1][0]))
            else:
                shap_dict = dict(zip(feature_names, shap_values[0]))

            shap_json = json.dumps({
                k: float(v[0]) if isinstance(v, np.ndarray) else float(v)
                for k, v in shap_dict.items()
            })

            # Log to database
            log = PredictionLog(
                Diabetes_Status=input_data[0],
                A1c_Percent=input_data[1],
                eGFR=input_data[2],
                Urine_Microalbumin_mg_L=input_data[3],
                Serum_Creatinine_umol_L=input_data[4],
                Fasting_Venous_Cholesterol_mmol_L=input_data[5],
                Age=input_data[6],
                Ave_SBP_mmHg=input_data[7],
                BMI_kg_m2=input_data[8],
                LDL_mmol_L=input_data[9],
                Risk_Score=risk_score,
                Risk_Category=risk_category
            )
            

            db.session.add(log)
            db.session.commit()

            return render_template("predict.html",
                                   risk_score=round(risk_score, 3),
                                   risk_category=risk_category,
                                   shap_data=shap_json)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("predict.html")

@app.route("/history")
def history():
    logs = PredictionLog.query.order_by(PredictionLog.Prediction_Time.desc()).all()
    return render_template("history.html", logs=logs)

@app.route("/download-history")
def download_history():
    logs = PredictionLog.query.order_by(PredictionLog.Prediction_Time.desc()).all()
    if not logs:
        return "No data to export."

    data = [{
         "ID": log.id,
        "Prediction Time": log.Prediction_Time.strftime("%Y-%m-%d %H:%M:%S"),
        "Diabetes Status": log.Diabetes_Status,
        "A1c Percent": log.A1c_Percent,
        "eGFR": log.eGFR,
        "Urine Microalbumin (mg/L)": log.Urine_Microalbumin_mg_L,
        "Serum Creatinine (umol/L)": log.Serum_Creatinine_umol_L,
        "Fasting Cholesterol (mmol/L)": log.Fasting_Venous_Cholesterol_mmol_L,
        "Age": log.Age,
        "Average SBP (mmHg)": log.Ave_SBP_mmHg,
        "BMI (kg/m²)": log.BMI_kg_m2,
        "LDL (mmol/L)": log.LDL_mmol_L,
        "Risk Score": round(log.Risk_Score, 3),
        "Risk Category": log.Risk_Category
    } for log in logs]

    df = pd.DataFrame(data)
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Prediction History', index=False)

    output.seek(0)
    return send_file(output, download_name="prediction_history.xlsx", as_attachment=True)

# Run the app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
