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
df = ps.read_csv(income_adult)
df.head()

sdf = spark.read.csv(income_adult, header=True, inferSchema=True).cache()
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

feature_numeric = ['age', 'fnlwgt', 'educational-num','capital-gain','capital-loss','hours-per-week']

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

#LOGISTIC REGRESSION

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol='income_indexed', featuresCol='features')

train, test = spark.read.csv(income_adult, header=True, inferSchema=True) \
     .cache() \
     .randomSplit([0.7, 0.3], seed = 5)

train.cache(), test.cache()

model = Pipeline(stages= [indexer_feature]+[indexer_label]+[encoders]+[assembler]+[lr]).fit(train)
pred_lr = model.transform(test)
pred_lr.select('prediction', 'income_indexed', 'features').show()
pred_lr.show(n=1, vertical=True, truncate=False)

#matrice de confusion
preds_and_labels = pred_lr.select(['prediction','income_indexed']).withColumn('label', F.col('income_indexed').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())

metrics.accuracy

#DECISION TREE CLASSIFIER
from pyspark.ml.classification import DecisionTreeClassifier

evaluator = MulticlassClassificationEvaluator(labelCol='income_indexed', predictionCol='prediction', metricName='accuracy')


tree = DecisionTreeClassifier(labelCol='income_indexed', featuresCol='features')

model = Pipeline(stages= [indexer_feature]+[indexer_label]+[encoders]+[assembler]+[tree]).fit(train)
pred_tree = model.transform(test)
pred_tree.select('prediction', 'income_indexed', 'features').show()

#matrice de confusion
preds_and_labels = pred_tree.select(['prediction','income_indexed']).withColumn('label', F.col('income_indexed').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())

accuracy_tree = evaluator.evaluate(pred_tree)
print("Accuracy of DecisionTree is = %g"% (accuracy_tree))
print("Error of DecisionTree = %g " % (1.0 - accuracy_tree))

#NAIVE BAYEs

from pyspark.ml.classification import NaiveBayes

#Toutes les features sont indépendantes
#Naive Bayes repose sur un calcul de proba
#Il va calculer la proba qu'une prédiction soit vraie la proba qu'elle soit fausse
#En comparant les 2 résultats on peut déduire à quelle classe appartient la donnée

nb = NaiveBayes(labelCol="income_indexed", featuresCol="features")
nb_model = Pipeline(stages= [indexer_feature]+[indexer_label]+[encoders]+[assembler]+[nb]).fit(train)
nb_prediction = nb_model.transform(test)
nb_prediction.select("prediction", "income_indexed", "features").show()

#matrice de confusion
preds_and_labels = nb_prediction.select(['prediction','income_indexed']).withColumn('label', F.col('income_indexed').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())

nbaccuracy = evaluator.evaluate(nb_prediction) 
print("Test accuracy = " + str(nbaccuracy))

#SVM
from pyspark.ml.classification import LinearSVC
svm = LinearSVC(labelCol="income_indexed", featuresCol="features")
svm_model = Pipeline(stages= [indexer_feature]+[indexer_label]+[encoders]+[assembler]+[svm]).fit(train)
svm_prediction = svm_model.transform(test)
svm_prediction.select("prediction", "income_indexed", "features").show()

#matrice de confusion
preds_and_labels = svm_prediction.select(['prediction','income_indexed']).withColumn('label', F.col('income_indexed').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())

svm_accuracy = evaluator.evaluate(svm_prediction)
print("Test accuracy = " + str(nbaccuracy)) 

# Gradient Boosted Tree

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="income_indexed", featuresCol="features",maxIter=10)
gbt_model = Pipeline(stages= [indexer_feature]+[indexer_label]+[encoders]+[assembler]+[gbt]).fit(train)
gbt_prediction = gbt_model.transform(test)


#matrice de confusion
preds_and_labels = gbt_prediction.select(['prediction','income_indexed']).withColumn('label', F.col('income_indexed').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())

gbt_accuracy = evaluator.evaluate(gbt_prediction)

print("Test accuracy = " + str(nbaccuracy))
