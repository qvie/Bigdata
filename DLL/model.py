from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# 1. Khởi tạo Spark Session
spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

# 2. Đọc dữ liệu
file_path = "diabetes_prediction_dataset.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True).dropna()
    
# 3. Tiền xử lý dữ liệu
# Chuyển đổi cột phân loại thành số
for col_name in ["gender", "smoking_history"]:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
    df = indexer.fit(df).transform(df)

# Chọn các cột đặc trưng và gộp thành một vector feature
feature_cols = ['gender_index', 'age', 'hypertension', 'heart_disease', 'smoking_history_index', 'bmi', 'HbA1c_level', 'blood_glucose_level']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Chỉ giữ lại cột cần thiết
final_df = df.select("features", col("diabetes").alias("label"))

# 4. Chia dữ liệu thành tập huấn luyện và kiểm tra
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

# 5. Huấn luyện mô hình Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=150, maxDepth=10, featureSubsetStrategy="auto")
model = rf.fit(train_data)

# 6. Dự đoán trên tập kiểm tra
predictions = model.transform(test_data)

# 7. Đánh giá mô hình
metrics = {
    "Accuracy": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions),
    "Precision": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions),
    "Recall": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions),
    "F1 Score": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions),
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 8. In ma trận nhầm lẫn
prediction_and_labels = predictions.select("prediction", "label").rdd.map(lambda row: (float(row.prediction), float(row.label)))
metrics = MulticlassMetrics(prediction_and_labels)
print("\nConfusion Matrix:")
print(metrics.confusionMatrix().toArray().astype(int))

# 9. Lưu mô hình
model.write().overwrite().save("diabetes_prediction_rf_model")

# Dừng Spark Session
spark.stop()
