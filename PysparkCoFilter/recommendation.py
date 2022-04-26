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
def cleanData(rawDataPath, reducedDataPath):
    print("|-- Start Clean Data...")
    trans = pd.read_csv(rawDataPath)
    # trans.rename({"rating": "count"}, axis="column", inplace=True)
    customArticlePair = trans[["customer_id", "article_id"]]
    countPlaceholder = pd.Series([1 for x in range(len(customArticlePair))])
    customArticlePair.loc[:, "count"] = countPlaceholder # Add a column of purchase count
    countGroup = customArticlePair.groupby(["customer_id", "article_id"]).count() # count purchase numbers
    countGroup.to_csv(reducedDataPath) # Save Counts to csv file

# Map string id to continuous int id
def mapData(reducedDataPath, mappedDataPath, customerDecPath, articleDecPath):
    print("|-- Start id Remapping...")
    countDf = pd.read_csv(reducedDataPath)
    index_encode(countDf, 'customer_id', customerDecPath)
    index_encode(countDf, 'article_id', articleDecPath)
    header = ["customer_id", "article_id", "count"]
    countDf.to_csv(mappedDataPath, header=header, index=False) # Save Counts to csv file

def sparkInit():
    print("|-- Initialize Spark...")
    conf = SparkConf()
    conf.setAll([('spark.app.name', 'Capstone Project'), ("spark.ui.port", "4621"),
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
def normData(ss, mappedDataPath, normDataPath):
    print("|-- Start Calculating average count...")
    inputParsed = ss.read.options(header=True, inferSchema='True',delimiter=',').csv(mappedDataPath).rdd
    countRDD = inputParsed.map(lambda p: Row(customer_id=int(p[0]), article_id=int(p[1]), count=int(p[2])))
    countDf = ss.createDataFrame(countRDD)
    countMean = countDf.groupby('customer_id').mean("count")
    countDf = countDf.join(countMean, on="customer_id", how="left")
    countDf = countDf.withColumn("countNorm", countDf["count"] - countDf["avg(count)"]) # Calculate average count per user
    countDf.select(col("customer_id"),col("article_id"), col("countNorm")).rdd.saveAsPickleFile(normDataPath)

def findBestModel(ss, normDataPath, modelPath):
    print("|-- Start Training Model...")
    pickle_rdd = ss.sparkContext.pickleFile(normDataPath).collect()
    schema = StructType([
        StructField("customer_id",IntegerType(),True),
        StructField("article_id",IntegerType(),True),
        StructField("countNorm", FloatType(), True)])
    countDf = ss.createDataFrame(pickle_rdd, schema=schema)
    (train, test) = countDf.randomSplit([0.8, 0.2])
    
    als = ALS(userCol="customer_id", itemCol="article_id", ratingCol="countNorm")
    param_grid = ParamGridBuilder().addGrid(als.rank, [12]).addGrid(als.regParam, [.05]).build() # .01, .1, .15
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="countNorm", predictionCol="prediction") 
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    model = cv.fit(train)
    best_model = model.bestModel
    best_model.setColdStartStrategy("drop")
    best_predictions = best_model.transform(test)
    rmse = evaluator.evaluate(best_predictions)
    print(f"The RMSE For the best Model is {rmse}")
    best_model.write().overwrite().save(modelPath)


"""
Use Model to recommend
"""
def recommend(modelPath, recRawPath):
    print("|-- Start Recommending...")
    best_model = ALSModel.load(modelPath)
    rec12Raw = best_model.recommendForAllUsers(12)
    rec12 = rec12Raw.withColumn("rec_exp", explode("recommendations"))\
    .select('customer_id', col("rec_exp.article_id"), col("rec_exp.rating"))

    w = Window.partitionBy(rec12.customer_id).orderBy(rec12.rating)
    rec_ranked = rec12.withColumn("rank", rank().over(w))
    rec_list = rec_ranked.groupBy("customer_id") .agg(collect_list("article_id").alias("articleList"))
    rec_dict = rec_list.toPandas()
    pickle_dump(recRawPath, rec_dict)
    
"""
Clean up results for submission
"""
def _write_formatted(df, fp, header, mode="w+"):
    df["stringTemp"] = df["articleList"].apply(lambda x: str(x)[1:-1].replace(',', ""))
    df[["customer_id", "stringTemp"]].to_csv(fp, header=header, index=None, sep=',', mode=f'{mode}')


def finalCleanup(recRawPath, customerDecPath, articleDecPath, recFinalPath):
    print("|-- Start Final Cleanup...")
    rec_dict = pickle_load(recRawPath)
    customerDec = pickle_load(customerDecPath)
    articleDec = pickle_load(articleDecPath)

    rec_dict["customer_id"] = rec_dict["customer_id"].map(customerDec)
    rec_dict["articleList"] = rec_dict["articleList"].apply(lambda x: list(itemgetter(*x)(articleDec)))
    _write_formatted(rec_dict, recFinalPath, header=["customer_id","prediction"])

"""
Generate Rec for missing users
"""
def fillEmpty(customerPath, rawDataPath, recFinalPath):
    print("|-- Filling Missing Users...")
    allCustomer = pd.read_csv(customerPath)["customer_id"]
    recCustomer = pd.read_csv(recFinalPath)["customer_id"]
    missing = list(set(allCustomer.unique()) ^ set(recCustomer.unique()))
    missingCustomerDf = pd.DataFrame (missing, columns = ['customer_id'])
    # Get Hottest Items
    transactions_train = pd.read_csv(rawDataPath, dtype={'customer_id': np.str})
    valueCnt = transactions_train['article_id'].value_counts().to_frame('count')
    top12Item = list(valueCnt.nlargest(12, "count").index)
    # Assign hottest item to missing user
    missingCustomerDf["articleList"] = np.repeat([top12Item], missingCustomerDf.shape[0], axis=0).tolist()
    _write_formatted(missingCustomerDf, recFinalPath, header=False, mode="a")
    
if __name__=="__main__":
    dataPath = "./co_filter"
    rawDataPath = f"{dataPath}/transactions_selected.csv"
    customerPath = f"{dataPath}/customers_clean.csv"
    reducedDataPath = f"{dataPath}/reducedData.csv"
    mappedDataPath = f"{dataPath}/mappedData.csv"
    customerDecPath = f"{dataPath}/customerDec.pkl"
    articleDecPath = f"{dataPath}/articleDec.pkl"
    normDataPath = f"{dataPath}/normData.pkl"
    modelPath = f"{dataPath}/savedModel/"
    recRawPath = f"{dataPath}/recRawData.pkl"
    fullRecPath = f"{dataPath}/fullRecData.csv"
    recFinalPath= f"{dataPath}/recommend12.csv"
    
    if not os.path.exists(reducedDataPath):
        cleanData(rawDataPath, reducedDataPath)
    if not os.path.exists(mappedDataPath):
        mapData(reducedDataPath, mappedDataPath, customerDecPath, articleDecPath)
    ss = sparkInit()
    if not os.path.exists(os.path.join(normDataPath, "_SUCCESS")):
        normData(ss, mappedDataPath, normDataPath)
    if not os.path.exists(modelPath):
        findBestModel(ss, normDataPath, modelPath)
    if not os.path.exists(recRawPath):
        recommend(modelPath, recRawPath)
#     if not os.path.exists(fullRecPath):
#         fillEmpty(customerPath, rawDataPath, recRawPath, fullRecPath)
    if not os.path.exists(recFinalPath):
        finalCleanup(recRawPath, customerDecPath, articleDecPath, recFinalPath)
        fillEmpty(customerPath, rawDataPath, recFinalPath)
        print(f"|-- Recommendation Generated for {pd.read_csv(recFinalPath).shape[0]} Users!")