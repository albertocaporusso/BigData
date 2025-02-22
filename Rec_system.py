from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# Setup Spark session
spark = SparkSession.builder.appName("BookRecommendation").getOrCreate()

# Load the data
df_books_name = pd.read_csv("BX-Books.csv", on_bad_lines='skip', sep=';', low_memory=False)
df_books_name = df_books_name.iloc[:, :-3]
df_books_name = df_books_name.set_index('ISBN')
df_books_name = df_books_name.rename_axis(None)

df_books_name_spark = spark.createDataFrame(df_books_name.reset_index())

df = pd.read_csv("BX-Book-Ratings.csv", on_bad_lines='skip', sep=';')
df = df[df['Book-Rating'] != 0]
df_spark = spark.createDataFrame(df)
df_spark = df_spark.filter(df_spark['ISBN'].isin([row['index'] for row in df_books_name_spark.select('index').collect()]))

# Filter and prepare the data
df_filter_spark = df_spark.groupBy('ISBN').count().filter('count >= 10').select('ISBN')
df_filter_spark = df_spark.join(df_filter_spark, on='ISBN')

users = df_filter_spark.select('User-ID').distinct().rdd.flatMap(lambda x: x).collect()
books = df_filter_spark.select('ISBN').distinct().rdd.flatMap(lambda x: x).collect()

schema = StructType([StructField('User-ID', IntegerType(), True)] + [StructField(book, FloatType(), True) for book in books])
df_books_spark = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

for row in df_filter_spark.collect():
    df_books_spark = df_books_spark.withColumn(row['ISBN'], F.when(col('User-ID') == row['User-ID'], row['Book-Rating']).otherwise(col(row['ISBN'])))

# Compute item-item similarity
df_books_pandas = df_books_spark.toPandas().fillna(0)
item_similarity = cosine_similarity(df_books_pandas.set_index('User-ID').T)

# User interactions and item scores
USER_ID = 104636
TOP = 3
user_interactions = df_books_pandas[df_books_pandas['User-ID'] == USER_ID].drop('User-ID', axis=1).values.flatten()
item_scores = user_interactions.dot(item_similarity)

item_scores[user_interactions > 0] = 0
item_scores = (item_scores - item_scores.min()) / (item_scores.max() - item_scores.min()) * 10
item_scores[user_interactions > 0] = 0

recommended_items = np.argsort(item_scores)[::-1][:TOP]

# Display recommended items
table_data = []
for item in recommended_items:
    Isnb = df_books_pandas.columns[item + 1]  # +1 because the first column is 'User-ID'
    title = df_books_name.loc[Isnb]
    Pred = item_scores[item]
    table_data.append([title['Book-Title'], title['Book-Author'], Isnb, Pred])

table_headers = ["Book Title", "Book Author", "ISBN", "Predicted Rating"]
print(tabulate(table_data, headers=table_headers))