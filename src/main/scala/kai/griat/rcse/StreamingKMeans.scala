package kai.griat.rcse

import Utils._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.scheduler.{StreamingListener, StreamingListenerBatchCompleted}
import org.apache.spark.streaming.{Milliseconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.StreamingKMeans

object StreamingKMeans {
    def main(args: Array[String]) {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf()
          .setAppName("Streaming K-means test")
          .setMaster("local[*]")

        val sc = new SparkContext(conf)
        sc.setLogLevel("ERROR")

        val ssc = new StreamingContext(sc, Milliseconds(1000))
        /*val trainingData = ssc.textFileStream(train_filename)
          .map(_.split(" "))
          .map(arr => arr.dropRight(1))
          .map(_.mkString("[", ",", "]"))
          .map(Vectors.parse)*/

        val trainingData = ssc.socketTextStream("localhost", 9999).map(_.split(" ")).map(_.mkString("[", ",", "]")).map(Vectors.parse)

        val testData = ssc.socketTextStream("localhost", 9998).map(l => Vectors.dense(l.toDouble))

        val numDimensions = 1
        val numClusters = 3
        val model = new StreamingKMeans()
          .setK(numClusters)
          .setHalfLife(1000, "points")
          //.setDecayFactor(0.0)
          .setRandomCenters(numDimensions, 0.0)

        val N = new StaticVar[Long](0L)
        val listener = new MyListener(model, N, sc)
        ssc.addStreamingListener(listener)

        model.trainOn(trainingData)

        model.predictOn(testData).print()

        ssc.start()
        ssc.awaitTermination()
    }
}

class StaticVar[T](var value: T)

class MyListener(model: StreamingKMeans, n: StaticVar[Long], sc: SparkContext) extends StreamingListener {
    override def onBatchCompleted(batchCompleted: StreamingListenerBatchCompleted) {
        if (batchCompleted.batchInfo.numRecords > 0) {
            n.value = n.value + batchCompleted.batchInfo.numRecords
            println("================= CENTERS ================= N = " + n.value)
            model.latestModel().clusterCenters.foreach(println)

            // Evaluate clustering by computing Within Set Sum of Squared Errors
            // (at the end of the training process)
            if (n.value == 1000) {
                evaluateWSSSE()
            }
        }
    }

    def evaluateWSSSE(): Unit = {
        val calcCostData = sc.textFile(train_filename)
          .map { line =>
              val parts = line.split(';')
              Vectors.dense(parts(0).split(' ').map(_.toDouble))
          }.cache()
        val WSSSE = model.latestModel.computeCost(calcCostData)
        println(s"Within Set Sum of Squared Errors = $WSSSE")
    }
}
