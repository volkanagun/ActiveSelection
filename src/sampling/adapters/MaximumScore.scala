package sampling.adapters

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class MaximumScore(val scorer:InstanceScorer, val topK:Int, maxSelectSize:Int) extends ScoreAdapter(maxSelectSize) {

  def filter(instances:Array[TextInstance]):Array[TextInstance] = {
    val scores = instances.map(textInstance => (scorer.score(textInstance), textInstance))
      .filter(pair=> !pair._1.isNaN && !pair._1.isInfinite)
    val topCount = maxSelectSize - count.toInt - 1
    val maximum = scores.sortBy(_._1).reverse
      .map(_._2)
      .take(math.min(topCount, topK))

    maximum.foreach(instance=> scorer.add(instance))
    scorer.postCompute()

    update(scores)
    maximum
  }

  override def init(array: Array[TextInstance]): MaximumScore.this.type = {
    scorer.init(array)
    this
  }
}
