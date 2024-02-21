package sampling.adapters

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.Random

class MovingAverage(val scorer:InstanceScorer, val k:Int, val kselect:Int, val stopSelect:Int, var threshold:Double) extends ScoreAdapter(stopSelect) {
  val random = new Random(42)
  var scores = Array[(Double, Double)]()

  override def filter(array: Array[TextInstance]): Array[TextInstance] = {
    measureTime {
      val collection = array.par
      collection.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(4))
      val instances = collection.map(instance => (scorer.score(instance), instance))
        .filter(pair => !pair._1.isNaN && !pair._1.isInfinite).toArray

      val m = mean()
      val s = std(m)

      val mapped = instances.map(p => (p._1, (p._1 - m) * math.sqrt(scores.length) / s, p._2))

      val selected = mapped
        .filter(_._2 > threshold)
        .sortBy(_._1).reverse
        .take(kselect)

      scores = scores ++ mapped.take(k).map(tuple=>(tuple._1,tuple._2))
      scores = scores.drop(scores.length - k)
      count += selected.size


      //threshold = scores.map(_._2).sum / scores.length

      val queried = selected.map(_._3)
      queried.foreach(textInstance => scorer.add(textInstance))
      scorer.postCompute()
      queried
    }
  }


  override def init(array: Array[TextInstance]): MovingAverage.this.type = {
    scorer.init(array)
    this
  }

  def scoring(array:Array[TextInstance]): this.type = {
    val instances = array.map(instance => (scorer.score(instance), 0.0))
      .filter(pair => !pair._1.isNaN && !pair._1.isInfinite).toArray

    scores = scores ++ instances
    this
  }

  def mean():Double={
    scores.map(_._1).sum / math.max(scores.length, 1)
  }

  def std(m:Double):Double = {
    math.sqrt(scores.map(_._1).map(s=> (s-m)*(s-m)).sum / (Math.max(1, scores.length-1))) + Double.MinPositiveValue
  }
}
