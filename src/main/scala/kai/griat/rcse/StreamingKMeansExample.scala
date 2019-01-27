package kai.griat.rcse

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streaming.{Seconds, StreamingContext}

object StreamingKMeansExample {

    val training_filename = "src/main/resources/prices.csv"
    val testing_filename = "src/main/resources/testing"

        def main(args: Array[String]): Unit = {

        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf()
          .setAppName("StreamingKMeansExample")
          .setMaster("local[2]")

        val sc = new SparkContext(conf)
        val ssc = new StreamingContext(sc, Seconds(4))

        val trainingData = ssc.textFileStream(training_filename)
          .map(line => Vectors.dense(line.split(' ').map(_.toDouble)))

        val testData = ssc
          .socketTextStream("localhost", 9999)
          .map(LabeledPoint.parse)

        val model = new StreamingKMeans()
          .setK(3)
          .setDecayFactor(1.0)
          .setRandomCenters(1, 1.0)

        model.trainOn(trainingData)
        model.predictOn(
            testData.map(lp =>
                lp.features
            )
        ).print()

        ssc.start()
        ssc.awaitTermination()

    }

}
