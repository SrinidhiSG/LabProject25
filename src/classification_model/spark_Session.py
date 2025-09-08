# Creating Spark Session
from pyspark.sql import SparkSession

def spark_session():
  spark = (SparkSession.builder.appName('Titanic MLOps Pipeline')
          .config('spark.executor.cores', '4')
          .config('spark.executor.instances', '2')
          .config('spark.executor.memory', '2g')
          .config('spark.driver.memory', '2g')
          .config('defaut.parallelism', '4')
          .config('spark.sql.shuffle.partitions', '4')
          .config("spark.hadoop.fs.file.impl.disable.cache", "true")
          .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs")
          .getOrCreate())
  return spark