from flask import Flask, request, render_template
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Khắc phục lỗi 'iteritems'
pd.DataFrame.iteritems = pd.DataFrame.items

app = Flask(__name__)
spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

# Load mô hình đã train
model_path = "diabetes_prediction_rf_model"
model = RandomForestClassificationModel.load(model_path)

global_predictions = None

# Tạo thư mục lưu trữ biểu đồ nếu chưa tồn tại
if not os.path.exists("static"): os.makedirs("static")

def preprocess_data(df):
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
    smoking_indexer = StringIndexer(inputCol="smoking_history", outputCol="smoking_index", handleInvalid="keep")

    df = gender_indexer.fit(df).transform(df)
    df = smoking_indexer.fit(df).transform(df)

    feature_cols = [
        "gender_index", "age", "hypertension", "heart_disease", 
        "smoking_index", "bmi", "HbA1c_level", "blood_glucose_level"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    return df

def generate_visualizations(df):
    # Biểu đồ phân phối tuổi
    plt.figure(figsize=(8, 6))
    sns.histplot(df['age'], bins=30, kde=True, color='blue')
    plt.title('Phân phối tuổi')
    plt.xlabel('Tuổi')
    plt.ylabel('Tần suất')
    plt.savefig('static/age_distribution.png', dpi=300)
    plt.close()

    # Biểu đồ BMI vs glucose
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='bmi', y='blood_glucose_level', hue='Prediction', data=df, palette='Set1')
    plt.title('BMI vs Mức đường huyết')
    plt.xlabel('BMI')
    plt.ylabel('Mức đường huyết')
    plt.savefig('static/bmi_vs_glucose.png', dpi=300)
    plt.close()

    # Biểu đồ phân phối giới tính
    plt.figure(figsize=(8, 6))
    sns.countplot(x='gender', hue='Prediction', data=df, palette='Set2')
    plt.title('Phân phối giới tính theo dự đoán')
    plt.xlabel('Giới tính')
    plt.ylabel('Số lượng')
    plt.savefig('static/gender_distribution.png', dpi=300)
    plt.close()

    # Biểu đồ HbA1c distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['HbA1c_level'], bins=30, kde=True, color='purple')
    plt.title('Phân phối HbA1c')
    plt.xlabel('HbA1c Level')
    plt.ylabel('Tần suất')
    plt.savefig('static/hba1c_distribution.png', dpi=300)
    plt.close()

    # Biểu đồ smoking vs prediction
    plt.figure(figsize=(8, 6))
    sns.countplot(x='smoking_history', hue='Prediction', data=df, palette='Set3')
    plt.title('Hút thuốc so với Dự đoán')
    plt.xlabel('Lịch sử hút thuốc')
    plt.ylabel('Số lượng')
    plt.savefig('static/smoking_vs_prediction.png', dpi=300)
    plt.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "smoking_history": request.form["smoking_history"],
            "bmi": float(request.form["bmi"]),
            "HbA1c_level": float(request.form["hba1c"]),
            "blood_glucose_level": float(request.form["glucose"])
        }

        df = spark.createDataFrame([input_data])
        df = preprocess_data(df)
        prediction = model.transform(df).select("prediction").collect()[0][0]

        result = "Có bệnh" if prediction == 1 else "Không có bệnh"
        return render_template("index.html", prediction_result=result)
    except Exception as e:
        return str(e)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    global global_predictions
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        spark_df = spark.createDataFrame(df)
        spark_df = preprocess_data(spark_df)
        predictions = model.transform(spark_df).select("prediction").toPandas()

        df["Prediction"] = predictions["prediction"].apply(lambda x: "Có bệnh" if x == 1 else "Không có bệnh")

        global_predictions = df
        # Tạo biểu đồ
        generate_visualizations(df)

        return render_template("visualization.html", charts=["age_distribution.png", "bmi_vs_glucose.png", "gender_distribution.png", "hba1c_distribution.png", "smoking_vs_prediction.png"], table=df.to_html(classes="table table-striped", index=False))

    except Exception as e:
        return str(e)

if __name__ == "__main__":  
    app.run(debug=True)
