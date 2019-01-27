package kai.griat.rcse.streamdm

import com.github.javacliparser.ClassOption
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streamdm.clusterers.Clustream
import org.apache.spark.streamdm.streams.StreamReader
import org.apache.spark.streaming.{Seconds, StreamingContext}

object CluStreamExample {

    val training_filename = "src/main/resources/prices.csv"
    val testing_filename = "src/main/resources/testing"

    val streamReaderOption: ClassOption = new ClassOption("streamReader", 's',
        "Stream reader to use", classOf[StreamReader], "SocketTextStreamReader")

    val clustererOption: ClassOption = new ClassOption("learner", 'l',
        "Learner to use", classOf[Clustream], "Clustream")

    def main(args: Array[String]): Unit = {

        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf()
          .setAppName("CluStream example")
          .setMaster("local[2]")

        val sc = new SparkContext(conf)
        val ssc = new StreamingContext(sc, Seconds(4))

        val trainingData = ssc.textFileStream(training_filename)
          .map(line => Vectors.dense(line.split(' ').map(_.toDouble)))

        val testData = ssc
          .socketTextStream("localhost", 9999)
          .map(LabeledPoint.parse)


        val reader: StreamReader = this.streamReaderOption.getValue()

        val clusterer: Clustream = this.clustererOption.getValue()
        clusterer.init(reader.getExampleSpecification())

        clusterer.microclusters.horizonOption.setValue(1)
        clusterer.initOption.setValue(2000)
        clusterer.kOption.setValue(5)
        clusterer.mcOption.setValue(50)


        //Parse stream and get Examples
        val N = new StaticVar[Long](0L)
        val listener = new MyListener(clusterer, N)
        ssc.addStreamingListener(listener)
        val instances = reader.getExamples(ssc)

        clusterer.train(instances)

    }
}
