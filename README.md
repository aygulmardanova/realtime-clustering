# realtime-clustering

Project for clustering methods comparing: 
* KMeans (MLlib)
* Streaming KMeans (MLlib, Spark Streaming)
* CluStream (StreamDM)

First, packaged streamDM.jar lib have to be added to module dependencies.
[StreamDM](http://huawei-noah.github.io/streamDM/) is a new open source software for mining big data streams using Spark Streaming. 

## Preparations
Java, Scala, Spark, SBT must be installed.

Clone StreamDM project from Github
From the main folder of StreamDM, enter the following command to generate the packages to run StreamDM: 
```
sbt package
```
### Current environment
* Spark 2.3.0
* Scala 2.11.8
* SBT 0.13.17
* Java 8