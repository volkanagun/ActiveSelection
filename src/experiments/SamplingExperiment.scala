package experiments

import evaluation._
import models.EmbeddingModel
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import sampling.experiments.SampleParams
import utils.{Params, Tokenizer}

import java.io.File
import scala.collection.parallel.CollectionConverters.ImmutableIterableIsParallelizable
import scala.util.Random
import scala.util.control.Breaks

class SamplingExperiment {

  val samplingNames = Array("VocabSelect", /*"VotedDivergence",*/ "KMeans","KL" ,"VE", "LM", "Mahalonabis", "Euclidean", /*"Entropy",*/ "Least", "Boltzmann", "Hopfield")
  //val samplingNames = Array("Mahalonabis")
  val adapterNames = Array("avg")

  val selectionSizes = Array(1000/*,5000, 25000*/)
  val tasks = Array(/*"ner","pos", "intrinsic",*/"sentiment")
  val models = Array(/*"cbow", "skip", */"self")
  val jsonFilename = "resources/evaluation/analogy/sentence-tr.json"

  val tokenizer = new Tokenizer(windowSize = 2).loadZip()
  val embedParams = new SampleParams()

  def experimentKey(params: Params): Int = {
    val extractorName = "feature"
    val embeddingSize = params.embeddingLength
    val hiddenSize = params.hiddenLength
    val clusterSize = params.clusterSize
    val windowSize = params.windowLength
    val committee = params.committee
    val knn = params.knn
    val tokenLength = params.tokenLength
    val dictionarySize = params.dictionarySize


    val array = Array[Int](
      params.maxSelectSize,
      dictionarySize,
      embeddingSize,
      hiddenSize,
      windowSize,
      clusterSize,
      tokenLength,
      true.hashCode(),
      committee,
      knn,
      extractorName.hashCode,
      params.selectionMethod.hashCode)

    val keyID = array.foldRight[Int](7) { case (crr, main) => main * 7 + crr }
    keyID
  }

  def evaluate(embedParams: SampleParams, model: String): Boolean = {
    if ("intrinsic".equals(embedParams.taskName)) evaluateIntrinsic(embedParams, model)
    else if ("ner".equals(embedParams.taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicNER(embedParams, tokenizer))
    else if ("pos".equals(embedParams.taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicPOS(embedParams, tokenizer))
    else if ("sentiment".equals(embedParams.taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicSentiment(embedParams, tokenizer))
    else false
  }

  def evaluateExtrinsic(embedParams: SampleParams, model: String, function: ExtrinsicSelf): Boolean = {

    val sentenceFilename = embedParams.sampledDataset()
    //embedParams.textFolder + function.getClassifier() +"/" + embedParams.selectionMethod + "-" + embedParams.maxSelectSize + "-" + key + ".txt"

    val modelling = model + "/" + embedParams.scorerName + "-" + embedParams.maxSelectSize + "/" + function.getClassifier() + "/" + embedParams.embeddingModel + "-" + embedParams.maxSelectSize + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Sentence filename: " + sentenceFilename + " Task: " + function.getClassifier())
      println("Result filename: " + resultName + " Task: " + function.getClassifier())
      val mainEvaluation = new IntrinsicEvaluation(resultName)
      mainEvaluation.functions :+= function

      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel, tokenizer)
        .train(sentenceFilename)

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
      true
    }
    else if (new File(resultName).exists()) {
      println("Found sentence: " + sentenceFilename)
      println("Found result: " + resultName)
      false
    }
    else {
      false
    }
  }


  def evaluateIntrinsic(embedParams: SampleParams, model: String): Boolean = {

    val sentenceFilename = embedParams.sampledDataset()
    //embedParams.intrinsicTextFolder + embedParams.scorerName + "-" + embedParams.maxSelectSize + "-" + key + ".txt"

    val modelling = model + "/" + embedParams.scorerName + "-" + embedParams.maxSelectSize + "/intrinsic/" + embedParams.embeddingModel + "-" + embedParams.maxSelectSize + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Result filename: " + resultName)
      val mainEvaluation = new IntrinsicEvaluation(resultName)
        .attachEvaluations(jsonFilename)
        .compile()

      mainEvaluation.filter(Array("SEMEVAL"))
      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel, tokenizer)
        .train(sentenceFilename)

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
      true
    }
    else if (new File(resultName).exists()) {
      println("Found: " + resultName)
      false
    }
    else {
      false
    }
  }


  def evaluate(mainEvaluation: IntrinsicEvaluation, model: EmbeddingModel, params: SampleParams): Unit = {

    println("==============================================================")
    println("==========" + params.embeddingModel.toUpperCase + "===============")
    println("==============================================================")

    mainEvaluation.evaluateReport(model, params)
  }
}

object SamplingExperiment {


  init()

  def main(args:Array[String]): Unit = {
    //System.setProperty("org.bytedeco.openblas.load", "mkl")
    val range = Range(0, 1).toArray//.par
    //range.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(8))
    range.foreach(_ =>
      evaluate(new SamplingExperiment, args)
    )
  }



  def init(): Unit = {
    Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.BFLOAT16)
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.BFLOAT16)
    //Nd4j.getMemoryManager.togglePeriodicGc(true)
    //Nd4j.getMemoryManager.setAutoGcWindow(50)
  }

  def adjustEpoc(size:Int):Int={
    if(size == 5000) 4
    else if(size == 1000) 5
    else if(size == 25000) 3
    else 2
  }


  def evaluate(experiments: SamplingExperiment, args: Array[String]): Unit = {
    val breaks = new Breaks()

    breaks.breakable {
      experiments.models.foreach(crrModel => {
        experiments.tasks.foreach(taskName => {
          experiments.selectionSizes.foreach(selectSize => {
            experiments.adapterNames.foreach(adapterName => {
              experiments.samplingNames.foreach(scorerName => {
                println(s"Evaluating ${scorerName} - ${selectSize} on task ${taskName}")
                val crrParams = experiments.embedParams.copy()
                crrParams.scorerName = scorerName
                crrParams.taskName = taskName
                crrParams.maxSelectSize = selectSize
                crrParams.adapterName = adapterName
                crrParams.embeddingModel = crrModel
                crrParams.epocs = adjustEpoc(selectSize)
                crrParams.batchSize = 128
                crrParams.evalBatchSize = 64
                crrParams.evalEpocs = 7
                crrParams.embeddingRandomMask = 1

                if (experiments.evaluate(crrParams, crrModel)) {
                  //breaks.break()
                }
              })
            })
          })
        })
      })
    }
  }
}
