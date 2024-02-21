package sampling.methods.statistical

import breeze.linalg
import breeze.linalg.DenseMatrix
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.inverse.InvertMatrix
import sampling.data.TextInstance
import smile.math.matrix.Matrix
import smile.math.matrix.Matrix.LU

import scala.util.Random

class MahalonabisScore(dictionarySize:Int, embeddingSize:Int) extends IDScorer(dictionarySize, embeddingSize) {


  var mean = Array.fill[Double](embeddingSize)(0d)
  var covariance = Array.fill[Array[Double]](embeddingSize)(Array.fill[Double](embeddingSize)(0.0))
  var covarianceInverse = Array[Array[Double]]()

  override def add(sample:TextInstance):this.type ={
    val array = embeddingVector(sample.featureSequence)
    addCovariance(array)

    mean = mean.zip(array).map(pair => pair._1 + pair._2 / count)
    count += 1
    this
  }


  override def postCompute(): MahalonabisScore.this.type = {
    val covMat = DenseMatrix.create(covariance.length, covariance.length, covariance.flatten)
    val inversed = linalg.pinv(covMat)
    covarianceInverse = inversed.toArray
      .sliding(covariance.length,covariance.length)
      .toArray
    this
  }



  def addCovariance(array:Array[Double]): this.type = {
    //val noise = new Random()
    array.zipWithIndex.filter(_._1 > 0).map{case(arri, iindex)=>{
      val meani = mean(iindex)
      val diffi = arri - meani
      array.zipWithIndex.filter(_._1 > 0).map{case(arrj, jindex)=>{
        val meanj = mean(jindex)
        val diffj = arrj - meanj
        covariance(iindex)(jindex) = covariance(iindex)(jindex) + (diffi * diffj)/count
      }}
    }}

    this
  }

  override def score(instance: TextInstance): Double = {
    val embeddings = embeddingVector(instance.featureSequence)
    val arr = covarianceInverse.map{ row =>{
      embeddings.zipWithIndex.map{case(xval, xi) => row(xi) * (xval-mean(xi))}.sum
    }}

    val total = math.sqrt(arr.map(item=> item * item).sum)
    1.0/total
  }


  override def init(instances: Array[TextInstance]): MahalonabisScore.this.type = {
    instances.foreach(instance=> add(instance))
    postCompute()
  }

}
