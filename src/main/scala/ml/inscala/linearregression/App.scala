package ml.inscala.linearregression

import org.apache.spark.{SparkConf,SparkContext}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.util.Try
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.Column


object App {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().master("local[*]").getOrCreate()

    val data = 
      spark.read.option("header","true")
      .option("inferSchema","true")
      .format("csv")
      .load("Clean-USA-Housing.csv")

    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors
    import spark.implicits._

    val df = 
      data
      .select(
        data("Price").as("label"),
        $"Avg Area Income",
        $"Avg Area House Age",
        $"Avg Area Number of Rooms",
        $"Avg Area Number of Bedrooms",
        $"Area Population"
      )

    val assembler = 
      new VectorAssembler()
      .setInputCols(Array(
        "Avg Area Income",
        "Avg Area House Age",
        "Avg Area Number of Rooms",
        "Avg Area Number of Bedrooms",
        "Area Population"
      ))
      .setOutputCol("features")


    val output = assembler.transform(df).select($"label",$"features")

    val lr = new LinearRegression()

    val lrModel = lr.fit(output)

    val trainingSummary = lrModel.summary

    trainingSummary.residuals.show()

    trainingSummary.predictions.show()

    println(trainingSummary.r2)

    spark.stop()
  }  
}