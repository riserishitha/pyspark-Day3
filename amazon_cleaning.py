from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark session
spark = SparkSession.builder \
    .appName("AmazonReviewCleaning") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "dataset/amazon_reviews.csv",
    header=True,
    inferSchema=True
)

print("Original Dataset")
df.show(5)
df.printSchema()

# -------------------------
# NULL HANDLING
# -------------------------

# Remove rows where review text is null
df_clean = df.dropna(subset=["Text"])

# Fill missing labels if any
df_clean = df_clean.fillna({"label": 0})

# -------------------------
# TYPE CASTING
# -------------------------

df_clean = df_clean.withColumn(
    "label",
    col("label").cast("int")
)

# -------------------------
# FILTERING
# -------------------------

# Keep only valid labels (0 or 1)
df_filtered = df_clean.filter(col("label").isin(0, 1))

print("Cleaned Dataset")
df_filtered.show(5)

# -------------------------
# WRITE TO PARQUET
# -------------------------

df_filtered.write \
    .mode("overwrite") \
    .parquet("output/amazon_reviews_parquet")

print("Dataset successfully written in Parquet format")

spark.stop()