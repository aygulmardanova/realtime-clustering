package kai.griat.rcse

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansExample {

    val def_filename = "src/main/resources/prices2.csv"
    val trained_models_dir = "output/trainedModels/KMeansModel"

    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        // Initialize spark configurations
        val conf = new SparkConf()
        .setAppName("example class")
        .setMaster("local[2]")

        // SparkContext
        val sc = new SparkContext(conf)

        // Load and parse the data
        val data = sc.textFile(def_filename)
        val parsedData = data.map { line =>
            val parts = line.split(';')
            Vectors.dense(parts(0).split(' ').map(_.toDouble))
        }.cache()

        // Split data into training and test
        val parsedDataSplits = parsedData.randomSplit(Array(0.3, 0.7), seed = 11L)

        val trainingData = parsedDataSplits(0)
        val testData = parsedDataSplits(1)

        // Cluster the data into classes using KMeans
        val numClusters = 5
        val numIterations = 20
        val clusters = KMeans.train(trainingData, numClusters, numIterations)

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = clusters.computeCost(parsedData)
        println(s"Within Set Sum of Squared Errors = $WSSSE")

        // Save and load model
        clusters.save(sc, trained_models_dir)
        val trainedModel = KMeansModel.load(sc, trained_models_dir)

        val filtered = testData.filter(v => trainedModel.predict(v) >= 0)
        filtered.foreach(v => println(v + " - " + trainedModel.predict(v)))
    }
}
