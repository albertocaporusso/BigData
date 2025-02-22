import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
import random, os, sys, time
import threading
# spark & ML
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import psutil
from pyspark.sql.types import IntegerType

def ram_usage_bar():
    try:
        while True:
            mem = psutil.virtual_memory()
            bar_length = 50
            used_mem_percentage = mem.percent
            bar = '█' * int(used_mem_percentage / 2) + '-' * (bar_length - int(used_mem_percentage / 2))
            sys.stdout.write(f'\rRAM Usage: [{bar}] {used_mem_percentage:.2f}%')
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping RAM usage monitor...")

def run():


    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    spark = SparkSession.builder.appName('rec-sys').config("spark.python.profile.memory","true").config("spark.driver.memory", "15g").config("spark.executor.memory", "15g").getOrCreate()
    ratings_df = spark.read.csv('BX-Book-Ratings.csv', sep=';',
                                inferSchema=True,header=True)
    books_df = spark.read.csv('BX-Books.csv', sep=';', inferSchema=True, header=True)
    books_df = books_df.drop('Image-URL-S', 'Image-URL-M', 'Image-URL-L')
    stringToInt = StringIndexer(inputCol='ISBN', outputCol='ISBN_int').fit(ratings_df)
    ratings_dfs = stringToInt.transform(ratings_df)
    ratings_df = ratings_dfs.filter(ratings_dfs['Book-Rating'] != 0)

    ratings_df = ratings_df.withColumn("ISBN_int", ratings_df["ISBN_int"].cast(IntegerType()))
    train_df, test_df = ratings_df.randomSplit([0.8,0.2])
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    rec_model = ALS( maxIter=20 ,regParam=0.35,userCol='User-ID',itemCol='ISBN_int',ratingCol='Book-Rating',
                nonnegative=True, coldStartStrategy="drop")

    print('Fitting model...')
    rec_model = rec_model.fit(train_df)
    predicted_ratings=rec_model.transform(test_df)
    evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction',labelCol='Book-Rating')
    rmse = evaluator.evaluate(predicted_ratings)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

if __name__ == '__main__':
    ram_thread = threading.Thread(target=ram_usage_bar)
    code_thread = threading.Thread(target=run)

    ram_thread.start()
    code_thread.start()

    code_thread.join()  # Wait for your code to finish
    print("\nYour code has completed.")
    ram_thread.join(timeout=1)  # Give some time for the
'''
predicted_ratings=rec_model.transform(test_df)
evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction',labelCol='Book-Rating')
rmse = evaluator.evaluate(predicted_ratings)
def recommend_for_user(user_id, n):
    ratings_user = ratings_dfs.filter(col('User-Id')==user_id)
    pred_ratings_user = rec_model.transform(ratings_user.filter(col('Book-Rating')==0))
    recs_user = books_df.join(pred_ratings_user.select(['ISBN', 'prediction']), on='ISBN')
    recs_user = recs_user.sort('prediction', ascending=False).drop('prediction').limit(n)
    return recs_user, pred_ratings_user
recs_user, pred_ratings_user = recommend_for_user(240567, 20)
pred_ratings_user.sort('prediction', ascending=False).show(20)'''