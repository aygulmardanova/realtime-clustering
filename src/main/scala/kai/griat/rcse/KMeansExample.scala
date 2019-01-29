package kai.griat.rcse

import Utils._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansExample {

    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        // Initialize spark configurations
        val conf = new SparkConf()
        .setAppName("example class")
        .setMaster("local[2]")

        // SparkContext
        val sc = new SparkContext(conf)

        // Load and parse the training and test data
        val trainingData = sc.textFile(train_filename)
          .map { line =>
            val parts = line.split(';')
            Vectors.dense(parts(0).split(' ').map(_.toDouble))
        }.cache()

        val testData = sc.textFile(test_filename)
          .map { line =>
              val parts = line.split(';')
              Vectors.dense(parts(0).split(' ').map(_.toDouble))
          }.cache()

        // Train model to cluster the data into classes using KMeans
        val numClusters = 3
        val numIterations = 20
        val clusters = KMeans.train(trainingData, numClusters, numIterations)

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = clusters.computeCost(trainingData)
        println(s"Within Set Sum of Squared Errors = $WSSSE")

        // Save and load model
        clusters.save(sc, trained_models_dir)
        val trainedModel = KMeansModel.load(sc, trained_models_dir)

        val filtered = testData.filter(v => trainedModel.predict(v) >= 0)
        filtered.foreach(v => println(v + " - " + trainedModel.predict(v)))
    }
}
