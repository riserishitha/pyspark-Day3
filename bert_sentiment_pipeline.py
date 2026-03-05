from pyspark.sql import SparkSession
from transformers import pipeline
import pandas as pd

# --------------------------------------
# 1. START SPARK SESSION
# --------------------------------------

spark = SparkSession.builder \
    .appName("AmazonReviews_BERT_Sentiment") \
    .getOrCreate()

print("Spark Session Started")

# --------------------------------------
# 2. LOAD CLEANED PARQUET DATA
# --------------------------------------

df = spark.read.parquet("output/amazon_reviews_parquet")

print("Dataset Loaded Successfully")
df.show(5)
df.printSchema()

# --------------------------------------
# 3. LIMIT DATA (SAFE FOR LOCAL MACHINE)
# --------------------------------------

df = df.select("Text").limit(200)

print("Using 200 rows for BERT inference")

# Convert Spark DataFrame → Pandas
pandas_df = df.toPandas()

print("Converted Spark DataFrame to Pandas")
print(pandas_df.head())

# --------------------------------------
# 4. LOAD BERT SENTIMENT MODEL
# --------------------------------------

print("Loading BERT model... (first run may download model)")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("BERT Model Loaded Successfully")

# --------------------------------------
# 5. RUN SENTIMENT PREDICTION
# --------------------------------------

def run_bert_prediction(text_list):
    results = sentiment_model(text_list)
    predictions = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    return predictions, scores


print("Running BERT sentiment predictions...")

preds, scores = run_bert_prediction(pandas_df["Text"].tolist())

pandas_df["bert_prediction"] = preds
pandas_df["confidence_score"] = scores

print("Predictions Completed")

print(pandas_df.head())

# --------------------------------------
# 6. CONVERT BACK TO SPARK DATAFRAME
# --------------------------------------

spark_predictions = spark.createDataFrame(pandas_df)

print("Converted Predictions Back to Spark DataFrame")

spark_predictions.show(10)

# --------------------------------------
# 7. SAVE FINAL DATASET AS PARQUET
# --------------------------------------

output_path = "output/amazon_reviews_with_bert"

spark_predictions.write \
    .mode("overwrite") \
    .parquet(output_path)

print("Final dataset saved to:", output_path)

# --------------------------------------
# 8. STOP SPARK SESSION
# --------------------------------------

spark.stop()

print("Pipeline Completed Successfully")