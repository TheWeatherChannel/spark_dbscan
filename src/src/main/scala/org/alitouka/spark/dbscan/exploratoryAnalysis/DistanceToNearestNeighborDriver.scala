package org.alitouka.spark.dbscan.exploratoryAnalysis

import org.apache.commons.math3.ml.distance.DistanceMeasure
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.alitouka.spark.dbscan.{RawDataSet, DbscanSettings}
import org.alitouka.spark.dbscan.util.commandLine._
import org.alitouka.spark.dbscan.util.io.IOHelper
import org.alitouka.spark.dbscan.util.debug.Clock
import org.alitouka.spark.dbscan.spatial.rdd.{PointsPartitionedByBoxesRDD, PartitioningSettings}
import org.alitouka.spark.dbscan.spatial.{DistanceCalculation, Point, PointSortKey, DistanceAnalyzer}

/** A driver program which estimates distances to nearest neighbor of each point
 *
 */
object DistanceToNearestNeighborDriver extends DistanceCalculation {

  private [dbscan] class Args extends CommonArgs with NumberOfBucketsArg with NumberOfPointsInPartitionArg

  private [dbscan] class ArgsParser
    extends CommonArgsParser (new Args (), "DistancesToNearestNeighborDriver")
    with NumberOfBucketsArgParsing [Args]
    with NumberOfPointsInPartitionParsing [Args]

  def main (args: Array[String]) {
    val argsParser = new ArgsParser()

    if (argsParser.parse(args)) {
      val clock = new Clock()

      val sc = new SparkContext(argsParser.args.masterUrl,
        "Estimation of distance to the nearest neighbor",
        jars = Array(argsParser.args.jar))

      run(
        sc,
        argsParser.args.inputPath,
        argsParser.args.outputPath,
        argsParser.args.distanceMeasure,
        argsParser.args.numberOfPoints,
        IOHelper.readDataset,
        IOHelper.saveTriples
      )

      clock.logTimeSinceStart("Estimation of distance to the nearest neighbor")
    }
  }

  def run(
    sc: SparkContext,
    inputPath: String,
    outputPath: String,
    distanceMeasure: DistanceMeasure,
    numberOfPoints: Long,
    reader: (SparkContext, String) => RawDataSet = IOHelper.readDataset,
    writer: (RDD[(Double, Double, Long)], String) => Unit = IOHelper.saveTriples) {

    val data = reader(sc, inputPath)
    val settings = new DbscanSettings().withDistanceMeasure(distanceMeasure)
    val partitioningSettings = new PartitioningSettings(numberOfPointsInBox = numberOfPoints)
    val partitionedData = PointsPartitionedByBoxesRDD (data, partitioningSettings)

    val pointIdsWithDistances = partitionedData.mapPartitions { it =>
      calculateDistancesToNearestNeighbors(it, settings.distanceMeasure)
    }

    val histogram = ExploratoryAnalysisHelper.calculateHistogram(pointIdsWithDistances)
    val triples = ExploratoryAnalysisHelper.convertHistogramToTriples(histogram)

    writer(sc.parallelize(triples), outputPath)
  }

  private [dbscan] def calculateDistancesToNearestNeighbors (
    it: Iterator[(PointSortKey, Point)],
    distanceMeasure: DistanceMeasure) = {

    val sortedPoints = it
      .map ( x => new PointWithDistanceToNearestNeighbor(x._2) )
      .toArray
      .sortBy( _.distanceFromOrigin )

    var previousPoints: List[PointWithDistanceToNearestNeighbor] = Nil

    for (currentPoint <- sortedPoints) {

      for (p <- previousPoints) {
        val d = calculateDistance(currentPoint, p)(distanceMeasure)

        if (p.distanceToNearestNeighbor > d) {
          p.distanceToNearestNeighbor = d
        }

        if (currentPoint.distanceToNearestNeighbor > d) {
          currentPoint.distanceToNearestNeighbor = d
        }
      }

      previousPoints = currentPoint :: previousPoints.filter {
        p => {
          val d = currentPoint.distanceFromOrigin - p.distanceFromOrigin
          p.distanceToNearestNeighbor >= d
        }
      }
    }

    sortedPoints.filter( _.distanceToNearestNeighbor < Double.MaxValue).map ( x => (x.pointId, x.distanceToNearestNeighbor)).iterator
  }
}
