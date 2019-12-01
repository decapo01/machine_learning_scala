
name         := "scala-ml-course"
organization := "ml.inscala.course"
version      := "1.0.0-SNAPSHOT"
scalaVersion := "2.12.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % "2.4.0",
  "org.apache.spark" %% "spark-sql"   % "2.4.0",
  "org.apache.spark" %% "spark-mllib" % "2.4.0",
)