from pyspark.sql import SparkSession
import pandas as ps
from pyspark.sql import functions as F
 # mean, col, split, col, regexp_extract, when, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pyspark import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark

income_adult = "adult.csv"    # full csv (not the cleaned csv) 
df = ps.read_csv(income)
df.head()

sdf = spark.read.csv(income, header=True, inferSchema=True).cache()
sdf.show()
sdf.is_cached            # Checks if df is cached
#  How many rows?
sdf.count()                # 11162
#  How many columns?
len(df.columns)

sdf.describe().toPandas()  # # par défaut ttes les col 
sdf.dtypes
sdf.select([col[0] for col in sdf.dtypes if col[1] != 'string']).describe().show()

df.describe()  # par défaut seulement les col numériques 
df.describe(include=['O']) # ça ne marche pas (comme ds le vrai Pandas) 
df_dtypes = df.dtypes
cols_cat = df_dtypes[df_dtypes == 'object'].index 
# cols_cat = ['job', 'marital','education','default','housing', 'loan', 'contact','month','poutcome','Target']
df[cols_cat].describe()

sdf.columns
labelCol = 'income'

feature_numeric = ['age', 'fnlwgt', 'education-num','capital-gain','capital-loss','hours-per-week']

feature_cat = ['workclass', 'education', 'marital-status','occupation', 'relationship','race', 'gender', 'native-country']
feature_cat_indexed = [col+'_indexed' for col in feature_cat]

feature_cat_encoded = [col +'_encoded' for col in feature_cat_indexed]
feature_cat_encoded

print(feature_cat_indexed)
    
print(feature_cat_encoded)


#STRING INDEXER
indexer_feature = StringIndexer(inputCols=feature_cat, handleInvalid='skip', outputCols=feature_cat_indexed)
indexer_label = StringIndexer(inputCol=labelCol, handleInvalid='skip', outputCol=labelCol+'_indexed')

sdf = indexer_feature.fit(sdf).transform(sdf)
#sdf.show(n=1, truncate=False, vertical=True)

#ONEHOTENCODER

encoders = OneHotEncoder(dropLast=False, inputCols=feature_cat_indexed, outputCols=feature_cat_encoded)  # handleInvalid='skip',  
sdf = encoders.fit(sdf).transform(sdf)
sdf.select(feature_cat_indexed+feature_cat_encoded).show(n=2, truncate=False, vertical=True)

#VECTORASSEMBLER

sdf = spark.read.csv(income_adult, header=True, inferSchema=True).cache()
assembler = VectorAssembler(inputCols=feature_cat_encoded+feature_numeric, outputCol='features')

#PIPELINE
Pipeline(stages= [indexer_feature]+[indexer_label]+[encoders]+[assembler]).fit(sdf).transform(sdf).show(n=1, truncate=False, vertical=True)

#SEPARATION ENTRE DONNE DE TEST ET TRAIN

train, test = sdf.randomSplit([0.7, 0.3],seed = 11)
train.show(n=1, truncate=False, vertical=True)
test.show(n=1, truncate=False, vertical=True)
