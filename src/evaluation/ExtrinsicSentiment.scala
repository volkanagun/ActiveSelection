package evaluation

import org.deeplearning4j.nn.graph.ComputationGraph
import sampling.experiments.SampleParams
import utils.Tokenizer

import scala.io.Source

class ExtrinsicSentiment(params:SampleParams, tokenizer: Tokenizer) extends ExtrinsicSelf(params, tokenizer, true){

  var categories :Array[String] = null
  var trainingSize = 5000

  init()

  override def getClassifier(): String = "sentiment"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/sentiment/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/sentiment/test.txt"
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {

    Source.fromFile(filename).getLines().filter(l=> l.contains("\t")).take(trainingSize).map(line=> {
      val mline = line.toLowerCase(locale)
      val array = mline.split("\t")
      val sentence = array.take(array.length - 1).mkString(" ")
      val label = array.last
      (sentence, label)
    })
  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if(categories == null){
      categories = loadSamples(getTraining()).map(_._2).toSet.toArray
    }

    categories
  }

  override def universe(): Set[String] = {
    Source.fromFile(getTraining()).getLines().filter(l=> l.contains("\t"))
      .take(trainingSize)
      .flatMap(sentence=> sentence.split("\t").head.split("\\s+"))
      .toSet
  }
}
