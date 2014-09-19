package org.alitouka.spark.dbscan.exploratoryAnalysis

import org.apache.commons.math3.ml.distance.DistanceMeasure
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.alitouka.spark.dbscan.util.io.IOHelper
import org.alitouka.spark.dbscan.{RawDataSet, DbscanSettings}
import ExploratoryAnalysisHelper._
import org.alitouka.spark.dbscan.util.commandLine._
import org.alitouka.spark.dbscan.spatial.DistanceAnalyzer
import org.alitouka.spark.dbscan.spatial.rdd.{PartitioningSettings, PointsPartitionedByBoxesRDD}
import org.alitouka.spark.dbscan.util.debug.Clock
import org.apache.spark.rdd.RDD

/** A driver program which calculates number of point's neighbors within specified distance
  * and generates a histogram of distribution of this number across the data set
  *
  */
object NumberOfPointsWithinDistanceDriver {

  private [dbscan] class Args extends CommonArgs with NumberOfBucketsArg with EpsArg with NumberOfPointsInPartitionArg

  private [dbscan] class ArgsParser
    extends CommonArgsParser(new Args(), "NumberOfPointsWithinDistanceDriver")
    with NumberOfBucketsArgParsing [Args]
    with EpsArgParsing[Args]
    with NumberOfPointsInPartitionParsing[Args]


  def main(args: Array[String]) {
    val argsParser = new ArgsParser()

    if (argsParser.parse(args)) {
      val clock = new Clock()
      val distance = argsParser.args.eps

      val sc = new SparkContext(argsParser.args.masterUrl,
        "Histogram of number of points within " + distance + " from each point",
        jars = Array(argsParser.args.jar))

      run(
        sc,
        distance,
        argsParser.args.distanceMeasure,
        argsParser.args.numberOfPoints,
        argsParser.args.numberOfBuckets,
        argsParser.args.inputPath,
        argsParser.args.outputPath,
        IOHelper.readDataset,
        IOHelper.saveTriples
      )

      clock.logTimeSinceStart("Calculation of number of points within " + distance)
    }
  }

  def run(
    sc: SparkContext,
    epsilon: Double,
    distanceMeasure: DistanceMeasure,
    numberOfPoints: Long,
    numberOfBuckets: Int,
    inputPath: String,
    outputPath: String,
    reader: (SparkContext, String) => RawDataSet = IOHelper.readDataset,
    writer: (RDD[(Double, Double, Long)], String) => Unit = IOHelper.saveTriples) {

    val data = reader(sc, inputPath)

    val settings = new DbscanSettings()
      .withEpsilon(epsilon)
      .withDistanceMeasure(distanceMeasure)

    val partitioningSettings = new PartitioningSettings(numberOfPointsInBox = numberOfPoints)

    val partitionedData = PointsPartitionedByBoxesRDD(data, partitioningSettings, settings)
    val distanceAnalyzer = new DistanceAnalyzer(settings)
    val closePoints = distanceAnalyzer.countClosePoints(partitionedData)

    val countsOfPointsWithNeighbors = closePoints
      .map(x => (x._1.pointId, x._2))
      .foldByKey (1L)(_+_)
      .cache()

    val indexedPoints = PointsPartitionedByBoxesRDD.extractPointIdsAndCoordinates(partitionedData)

    val countsOfPointsWithoutNeighbors = indexedPoints
      .keys
      .subtract(countsOfPointsWithNeighbors.keys)
      .map((_, 0L))

    val allCounts = countsOfPointsWithNeighbors union countsOfPointsWithoutNeighbors
    allCounts.persist()

    val histogram = ExploratoryAnalysisHelper.calculateHistogram(allCounts, numberOfBuckets)
    val triples: Seq[(Double, Double, Long)] = ExploratoryAnalysisHelper.convertHistogramToTriples(histogram)
    writer(sc.parallelize(triples), outputPath)
    allCounts.unpersist()
  }
}
