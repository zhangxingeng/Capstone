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

# Clean Parse Data
def cleanData(rawDataPath):
    print("|-- Start Clean Data...")
    trans = pd.read_csv(rawDataPath, dtype={'customer_id': str, 'article_id': str, 't_dat': str, 'price': float})
    customArticlePair = trans[["customer_id", "article_id"]].head(1000)
    countPlaceholder = pd.Series([1 for x in range(len(customArticlePair))])
    customArticlePair.loc[:, "count"] = countPlaceholder # Add a column of purchase count
    countDf = customArticlePair.groupby(["customer_id", "article_id"]).count() # count purchase numbers
    countDfReset = countDf.reset_index(level=["customer_id", "article_id"]) # Save Counts to csv file
    cleanedDf = countDfReset[["customer_id", "article_id", "count"]]
    return cleanedDf

def index_encode(df, colName):
    idList = df[colName].unique()
    idMap = list(range(len(idList)))
    encoder = dict(zip(idList, idMap))
    df[colName] = df[colName].map(encoder)
    decoder = dict(zip(idMap, idList))
    return decoder

def mapIndex(df):
    print("|-- Start id Remapping...")
    customerDecoder = index_encode(df, 'customer_id', )
    articleDecoder = index_encode(df, 'article_id', )
    return df, customerDecoder, articleDecoder

def sparkInit():
    print("|-- Initialize Spark...")
    conf = SparkConf()
    conf.setAll([('spark.app.name', 'Capstone Project'), ( "spark.sql.shuffle.partitions", 16),
                 ('spark.executor.memory', '16g'), ('spark.driver.memory','16g'), 
                 ('spark.executor.cores', '4'), ('spark.cores.max', '8')])
    
    ss = SparkSession.builder.config(conf=conf).getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    return ss

def normData(sparkDf):
    print("|-- Start Calculating average count...")
    countMean = sparkDf.groupby('customer_id').mean("count")
    normDf = sparkDf.join(countMean, on="customer_id", how="left")
    normDf = normDf.withColumn("countNorm", normDf["count"] / normDf["avg(count)"])
    return normDf[['customer_id', "article_id", "countNorm"]]

def findBestModel(normDf):
    print("|-- Start Training Model...")
    (train, test) = normDf.randomSplit([0.8, 0.2])
    als = ALS(maxIter=5, rank=12,regParam=0.01, userCol="customer_id", itemCol="article_id", ratingCol="countNorm", coldStartStrategy="drop")
    # param_grid = ParamGridBuilder()\
    # .addGrid(als.rank, [12])\
    # .addGrid(als.regParam, [.01])\
    # .build() # .01, .1, .15
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="countNorm", predictionCol="prediction") 
    # cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    model = als.fit(train)
    # best_model = model.bestModel
    best_predictions = model.transform(test)
    rmse = evaluator.evaluate(best_predictions)
    print(f"The RMSE For the best Model is {rmse}")
    return model

def recommend(model):
    rec12Raw = model.recommendForAllUsers(12)
    rec12Pd = rec12Raw.toPandas()
    sorter = lambda x: sorted(x, key = lambda t: t[1], reverse=True)
    spliter = lambda x: list(zip(*x))[0]
    rec12Pd["sorted"] = rec12Pd["recommendations"].apply(sorter)
    rec12Pd["recList"] = rec12Pd["sorted"].apply(spliter)
    return rec12Pd[["customer_id", "recList"]]

def _fillEmpty(customerPath, rawRec):
    print("|-- Filling Missing Users...")
    allCustomer = pd.read_csv(customerPath, dtype={'customer_id': str})["customer_id"]
    recCustomer = rawRec["customer_id"]
    missing = list(set(allCustomer.unique()) ^ set(recCustomer.unique()))
    missingCustomerDf = pd.DataFrame(missing, columns=['customer_id'])
    # Get Hottest Items
    transactions_train = pd.read_csv(rawDataPath, dtype={'article_id': str})
    valueCnt = transactions_train['article_id'].value_counts().to_frame('count')
    top12Item = list(valueCnt.nlargest(12, "count").index)
    # Assign hottest item to missing user
    missingCustomerDf["articleList"] = np.repeat([top12Item], missingCustomerDf.shape[0], axis=0).tolist()
    return missingCustomerDf

def mapDataBack(customerDecoder, articleDecoder, customerPath, rawRec):
    print("|-- Start Mapping Index Back...")
    listDecoder = lambda x: list(itemgetter(*x)(articleDecoder)) 
    rawRec["customer_id"] = rawRec["customer_id"].map(customerDecoder)
    rawRec["articleList"] = rawRec["recList"].apply(listDecoder)
    # print("!*********! RawRec Shape Before Fill: ", rawRec.shape[0])
    recMissing = _fillEmpty(customerPath, rawRec)
    recAll = pd.concat([rawRec[["customer_id", "articleList"]], 
                recMissing[["customer_id", "articleList"]]])
    return recAll

def writeFormatted(df, fp, header):
    df["stringTemp"] = df["articleList"].apply(lambda x: str(x)[1:-1].replace(',', "").replace('\'', ""))
    df[["customer_id", "stringTemp"]].to_csv(fp, header=header, index=None, sep=',', mode="w+")


if __name__=="__main__":
    dataPath = "./co_filter"
    rawDataPath = f"{dataPath}/transactions_train.csv"
    customerPath = f"{dataPath}/customers.csv"
    recFinalPath = f"{dataPath}/recommend12.csv"

    cleanedDf = cleanData(rawDataPath)
    mappedDf, customerDecoder, articleDecoder = mapIndex(cleanedDf)
    ss = sparkInit()
    sparkDf = ss.createDataFrame(mappedDf)
    normedDf = normData(sparkDf)
    model = findBestModel(normedDf)
    recommend = recommend(model)
    finalRec = mapDataBack(customerDecoder, articleDecoder, customerPath, recommend)
    writeFormatted(finalRec, recFinalPath, header=["customer_id", "prediction"])