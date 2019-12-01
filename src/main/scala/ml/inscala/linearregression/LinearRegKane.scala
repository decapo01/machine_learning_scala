package ml.inscala.linearregression
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression

object LinearRegKane {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.DEBUG)

    val spark =
      SparkSession
      .builder()
      .appName("LinRegDf")
      .master("local[*]")
      .getOrCreate()

    val inputLines = spark.sparkContext.textFile("regression.txt")

    val data = 
      inputLines
      .map(_.split(","))
      .map(x => (x(0).toDouble,Vectors.dense(x(1).toDouble)))


    import spark.implicits._


    val colNames = Seq("label","features")

    val df = data.toDF(colNames: _*)

    val trainTest = df.randomSplit(Array(0.5,0.5))

    val trainingDF = trainTest(0)

    val testDF = trainTest(1)

    val lir = new LinearRegression()

    val model = lir.fit(trainingDF)

    val fullPredictions = model.transform(testDF).cache()

    val predictionAndLabel = 
      fullPredictions
      .select("prediction","label")
      .rdd
      .map(x => (x.getDouble(0), x.getDouble(1)))

    for (prediction <- predictionAndLabel) {
      println(prediction)
    }

    spark.stop()
  }
}