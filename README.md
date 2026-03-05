# Amazon Reviews Sentiment Analysis Pipeline (PySpark + BERT)

## Project Overview

This project builds a **data pipeline that processes Amazon product reviews using PySpark and AI-based sentiment analysis**. The dataset is cleaned using PySpark and then analyzed using a **pretrained BERT transformer model** to determine whether reviews are **positive or negative**. The final processed dataset is stored in **Parquet format**, which is optimized for big data analytics.

---

# Technologies Used

* **Python**
* **PySpark** (Data Processing)
* **Transformers Library (BERT)** (AI / NLP)
* **PyTorch**
* **Parquet** (Efficient data storage)
* **VS Code**

---

# Dataset

The project uses an **Amazon Reviews dataset** containing customer review text and sentiment labels.

Example columns:

| Column | Description                                  |
| ------ | -------------------------------------------- |
| Text   | Customer review text                         |
| label  | Sentiment label (0 = negative, 1 = positive) |

---

# Pipeline Workflow

Raw Dataset
↓
Load Data using PySpark
↓
Data Cleaning (Null Handling, Filtering, Casting)
↓
Sentiment Analysis using BERT
↓
Add Predictions to Dataset
↓
Save Output as Parquet

---

# Key Steps

## 1. Data Loading

The Amazon review dataset is loaded into a **Spark DataFrame** for distributed processing.

## 2. Data Cleaning

Cleaning steps include:

* Removing null values
* Filtering empty reviews
* Casting data types

This ensures high-quality input for the AI model.

## 3. AI Sentiment Analysis

A **pretrained BERT model** from the Transformers library is used to classify reviews as:

* **POSITIVE**
* **NEGATIVE**

The model understands contextual meaning in text, improving sentiment prediction accuracy.

## 4. Save Results

The processed dataset with predictions is saved in **Parquet format**, which provides:

* Faster query performance
* Better compression
* Efficient big data storage

---

# Project Structure

```
pyspark-Day3/
│
├── dataset/
│   └── amazon_reviews.csv
│
├── output/
│   └── amazon_reviews_parquet/
│
├── amazon_cleaning.py
│
└── README.md
```

---

# How to Run

Activate environment:

```
spark-env\Scripts\activate
```

Install dependencies:

```
pip install pyspark transformers torch
```

Run the project:

```
python amazon_cleaning.py
```

---

# Skills Demonstrated

* Big Data Processing with **PySpark**
* NLP using **BERT Transformers**
* Data Cleaning & Preprocessing
* Building an **end-to-end data pipeline**
* Storing data using **Parquet**

---

# Conclusion

This project demonstrates how **distributed data processing with PySpark** can be combined with **transformer-based AI models** to analyze large-scale customer review data efficiently.

## Additional info 

pip install transformers
pip install torch
pip install pandas
pip install pyarrow


python -m pip install --upgrade pip

pip install pandas --trusted-host pypi.org --trusted-host files.pythonhosted.org

pip install transformers --trusted-host pypi.org --trusted-host files.pythonhosted.org

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install pandas pyarrow transformers