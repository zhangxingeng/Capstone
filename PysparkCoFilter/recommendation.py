import os
import pandas as pd
import numpy as np
import pickle
from operator import itemgetter

import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType,StructField, IntegerType,FloatType

# from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, explode, collect_list, rank
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# from pyspark.ml.evaluation import RegressionMetrics, RankingMetrics

def pickle_load(path):
    with open(path, 'rb+') as f:
        return pickle.load(f)

def pickle_dump(path, d):
    with open(path, 'wb+') as f:
        pickle.dump(d, f)

"""
Prep Raw Data for Model Training
"""
def index_encode(df, colName, decPath):
    idList = df[colName].unique()
    idMap = list(range(len(idList)))
    decodebook = dict(zip(idMap, idList))
    pickle_dump(decPath, decodebook)
    encodebook = dict(zip(idList, idMap))
    df[colName] = df[colName].map(encodebook)

# Clean Parse Data
def cleanData(rawDataPath):
    print("|-- Start Clean Data...")
    trans = pd.read_csv(rawDataPath, dtype={'customer_id': str, 'article_id': str})
    # trans.rename({"rating": "count"}, axis="column", inplace=True)
    customArticlePair = trans[["customer_id", "article_id"]] #.head(1000)
    countPlaceholder = pd.Series([1 for x in range(len(customArticlePair))])
    customArticlePair.loc[:, "count"] = countPlaceholder # Add a column of purchase count
    countDf = customArticlePair.groupby(["customer_id", "article_id"]).count() # count purchase numbers
    countDfReset = countDf.reset_index(level=["customer_id", "article_id"]) # Save Counts to csv file
    return countDfReset

# Map string id to continuous int id
def mapData(mappedDataPath, customerDecPath, articleDecPath, countDf):
    print("|-- Start id Remapping...")
    index_encode(countDf, 'customer_id', customerDecPath)
    index_encode(countDf, 'article_id', articleDecPath)
    header = ["customer_id", "article_id", "count"]
    countDf.to_csv(mappedDataPath, header=header, index=False) # Save Counts to csv file

def sparkInit():
    print("|-- Initialize Spark...")
    conf = SparkConf()
    conf.setAll([('spark.app.name', 'Capstone Project'), ( "spark.sql.shuffle.partitions", 16),
                 ('spark.executor.memory', '16g'), ('spark.driver.memory','16g'), 
                 ('spark.executor.cores', '4'), ('spark.cores.max', '8')])
    
    ss = SparkSession.builder.config(conf=conf).getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    return ss
    
    
"""
Load data and Find the best model
"""
# 
# Calculate Average
def normData(ss, mappedDataPath):
    print("|-- Start Calculating average count...")
    schema = StructType([
        StructField("customer_id",IntegerType(),True),
        StructField("article_id",IntegerType(),True),
        StructField("count", IntegerType(), True)]
    )
    normDf = ss.read.options().load(mappedDataPath, format="csv", delimiter=',', header=True, schema=schema)
    countMean = normDf.groupby('customer_id').mean("count")
    normDf = normDf.join(countMean, on="customer_id", how="left")
    normDf = normDf.withColumn("countNorm", normDf["count"] - normDf["avg(count)"]) # Calculate average count per user
    # normDf.select(col("customer_id"),col("article_id"), col("countNorm")).rdd.saveAsPickleFile(normDataPath)
    return normDf

def findBestModel(normDf):
    print("|-- Start Training Model...")
    (train, test) = normDf.randomSplit([0.8, 0.2])
    als = ALS(maxIter=5, userCol="customer_id", itemCol="article_id", ratingCol="countNorm", coldStartStrategy="drop")
    param_grid = ParamGridBuilder()\
    .addGrid(als.rank, [12])\
    .addGrid(als.regParam, [.01])\
    .build() # .01, .1, .15
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="countNorm", predictionCol="prediction") 
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    model = cv.fit(train)
    best_model = model.bestModel
    best_model.setColdStartStrategy("drop")
    best_predictions = best_model.transform(test)
    rmse = evaluator.evaluate(best_predictions)
    print(f"The RMSE For the best Model is {rmse}")
    return best_model
    # best_model.write().overwrite().save(modelPath)


"""
Use Model to recommend
"""
def recommend(model):
    print("|-- Start Recommending...")
    rec12Raw = model.recommendForAllUsers(12)
    rec12 = rec12Raw.withColumn("rec_exp", explode("recommendations"))\
    .select('customer_id', col("rec_exp.article_id"), col("rec_exp.rating"))

    w = Window.partitionBy(rec12.customer_id).orderBy(rec12.rating)
    rec_ranked = rec12.withColumn("rank", rank().over(w))
    rec_list = rec_ranked.groupBy("customer_id") .agg(collect_list("article_id").alias("articleList"))
    rec_dict = rec_list.toPandas()
    return rec_dict
    

"""
Generate Rec for missing users
"""
def _fillEmpty(customerPath, rawRec):
    print("|-- Filling Missing Users...")
    allCustomer = pd.read_csv(customerPath)["customer_id"]
    recCustomer = rawRec["customer_id"]
    missing = list(set(allCustomer.unique()) ^ set(recCustomer.unique()))
    missingCustomerDf = pd.DataFrame (missing, columns = ['customer_id'])
    # Get Hottest Items
    transactions_train = pd.read_csv(rawDataPath, dtype={'customer_id': str})
    valueCnt = transactions_train['article_id'].value_counts().to_frame('count')
    top12Item = list(valueCnt.nlargest(12, "count").index)
    # Assign hottest item to missing user
    missingCustomerDf["articleList"] = np.repeat([top12Item], missingCustomerDf.shape[0], axis=0).tolist()
    rawRec.append(missingCustomerDf, ignore_index = True)


def finalCleanup(customerDecPath, articleDecPath, customerPath, rawRec):
    print("|-- Start Final Cleanup...")
    customerDec = pickle_load(customerDecPath)
    articleDec = pickle_load(articleDecPath)

    rawRec["customer_id"] = rawRec["customer_id"].map(customerDec)
    rawRec["articleList"] = rawRec["articleList"].apply(lambda x: list(itemgetter(*x)(articleDec)))
    _fillEmpty(customerPath, rawRec)
    pickle_dump("./co_filter/finalRecJIC.pd", rawRec)
    return rawRec

"""
Clean up results for submission
"""
def writeFormatted(df, fp, header):
    df["stringTemp"] = df["articleList"].apply(lambda x: str(x)[1:-1].replace(',', ""))
    df[["customer_id", "stringTemp"]].to_csv(fp, header=header, index=None, sep=',', mode="w+")

    
if __name__=="__main__":
    dataPath = "./co_filter"
    rawDataPath = f"{dataPath}/transactions_selected.csv"
    customerPath = f"{dataPath}/customers_clean.csv"
    mappedDataPath = f"{dataPath}/mappedData.csv"
    customerDecPath = f"{dataPath}/customerDec.pkl"
    articleDecPath = f"{dataPath}/articleDec.pkl"
    recFinalPath= f"{dataPath}/recommend12.csv"
    
    countDf = cleanData(rawDataPath)
    mapData(mappedDataPath, customerDecPath, articleDecPath, countDf)
    
    ss = sparkInit()
    normDf = normData(ss, mappedDataPath)
    bestModel = findBestModel(normDf)
    rawRec = recommend(bestModel)
    finalRec = finalCleanup(customerDecPath, articleDecPath, customerPath, rawRec)
    writeFormatted(finalRec, recFinalPath, header=["customer_id","prediction"])
    print(f"|-- Recommendation Generated for {pd.read_csv(recFinalPath).shape[0]} Users!")
