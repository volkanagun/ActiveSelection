package models

import evaluation.IntrinsicFunction
import org.deeplearning4j.nn.graph.ComputationGraph
import sampling.experiments.SampleParams
import utils.Tokenizer

import java.util.Locale

abstract class EmbeddingModel(val params: SampleParams, val tokenizer:Tokenizer, var isEvaluation:Boolean = false) extends IntrinsicFunction {

  var avgTime = 0d
  var sampleCount = 0
  var locale = new Locale("tr")
  var dictionaryIndex = Map[String, Int]("dummy" -> 0)
  var dictionary = Map[String, Array[Float]]("dummy" -> Array.fill[Float](params.embeddingLength)(0f))

  var computationGraph: ComputationGraph = null



  def getTrainTime(): Double = avgTime

  def train(filename: String): EmbeddingModel

  def save(): EmbeddingModel

  def load(): EmbeddingModel


  def getDictionary(): Map[String, Array[Float]] = dictionary

  def getDictionaryIndex(): Map[Int, Array[Float]] = {
    dictionary.map { case (ngram, vector) => dictionaryIndex(ngram) -> vector }
  }

  def update(ngram: String, vector: Array[Float]): Int = {
    dictionary = dictionary.updated(ngram, vector)
    update(ngram)
  }

  def update(ngram: String): Int = {

    if (dictionaryIndex.size < params.evalDictionarySize) {
      dictionaryIndex = dictionaryIndex.updated(ngram, dictionaryIndex.getOrElse(ngram, dictionaryIndex.size))
    }
    retrieve(ngram)

  }
  def update(ngram: String, dictionarySize:Int): Int = {

    if (dictionaryIndex.size < dictionarySize) {
      dictionaryIndex = dictionaryIndex.updated(ngram, dictionaryIndex.getOrElse(ngram, dictionaryIndex.size))
    }
    retrieve(ngram)

  }

  def retrieve(ngram: String): Int = {
    dictionaryIndex.getOrElse(ngram, 0)
  }

  def tokenize(sentence: String): Array[String] = {
    val lwSentence = sentence.toLowerCase(locale)
    val ngrams = tokenizer.ngramFilter(lwSentence)
    val result = ngrams
      .flatMap(ngram=> ngram.split("\\s+"))

    result
  }

  def forward(token: String): Array[Float] = {
    val lwtoken = token.toLowerCase(locale)
    val frequentNgrams = tokenizer.ngramStemFilter(lwtoken)
      .flatMap(ngram=> ngram.split("\\s+"))
      .filter(ngram => dictionary.contains(ngram))

    val ngramVectors = frequentNgrams.map(ngram => dictionary(ngram))
    average(ngramVectors)
  }

  def average(embeddings: Array[Array[Float]]): Array[Float] = {
    var foldResult = Array.fill[Float](params.embeddingLength)(0f)
    embeddings.foldRight[Array[Float]](foldResult) { case (a, main) => {
      main.zip(a).map(pair => pair._1 + pair._2)
    }
    }
  }
}
