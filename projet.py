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