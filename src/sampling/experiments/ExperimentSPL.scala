package sampling.experiments

import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse
import sampling.adapters.{DirectSelect, MajorityVoting, MovingAverage, ScoreAdapter}
import sampling.data.TextInstance
import sampling.methods.clustering.{KMeansScorer, LanguageModelScorer}
import sampling.methods.committee.{KLScorer, VEScorer, VotedDivergence}
import sampling.methods.core._
import sampling.methods.nn.{BoltzmannScorer, HopfieldScorer, LeastSquares}
import sampling.methods.statistical._
import utils.Tokenizer

import java.io.{File, FileOutputStream, PrintWriter}
import java.util.concurrent
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport

import scala.io.Source
import scala.util.control.Breaks

object ExperimentSPL {

  val samplingNames = Array("KMeans", "VocabSelect", "VotedDivergence",  "VE", "LM", "Mahalonabis", "Euclidean", "Entropy","Hopfield",  "KL", "Least", "Boltzmann")
  val taskNames = Array("intrinsic", "ner", "pos",  "sentiment")

  //val samplingNames = Array("Boltzmann")
  val tokenization = Array("feature")
  val adapterNames = Array("avg")
  val selectionSizes = Array(1000, 5000, 25000)
  val embedParams = new SampleParams()

  embedParams.committee = 10
  embedParams.clusterSize = 20
  embedParams.windowSize = 20
  embedParams.kmoving = 200
  embedParams.kselectSize = 10

  embedParams.freqcutoff = 10
  embedParams.tokenLength = 5
  embedParams.threshold = 0.015
  embedParams.ngramCombinationSize = 10
  embedParams.useWords = true
  embedParams.maxSelectSize = embedParams.max100thausand
  embedParams.maxSentenceSize = embedParams.max100million
  embedParams.maxWordSamples = 10000
  embedParams.batchSize = 100
  embedParams.dictionarySize = 100000
  embedParams.secondDictionarySize = 5000
  embedParams.hiddenSize = 20
  embedParams.embeddingSize = 300
  embedParams.embeddingWindowSize = 10

  def createAdapter(adapterName: String, scorer: InstanceScorer, k: Int, kselectSize: Int, maxSelectSize: Int, threshold: Double): ScoreAdapter = {
    if ("avg".equals(adapterName) && scorer.isInstanceOf[VoabSelect]) new DirectSelect(maxSelectSize)
    else if ("avg".equals(adapterName)) new MovingAverage(scorer, k, kselectSize, maxSelectSize, threshold)
    else {
      val criteriaScorers = Array("VotedDivergence", "Hopfield", "VE", "LM", "Mahalonabis", "Euclidean", "Entropy", "KMeans", "Boltzmann", "KL", "Least").map(createCriterias(_))
      new MajorityVoting(criteriaScorers, k, kselectSize, maxSelectSize, threshold)
    }
  }

  def createCriterias(sampleName: String): InstanceScorer = {

    if ("KMeans".equals(sampleName)) {
      new KMeansScorer(embedParams.secondDictionarySize, embedParams.embeddingSize, embedParams.clusterSize, embedParams.knn)
    }
    else if ("KL".equals(sampleName)) {
      new KLScorer(embedParams.dictionarySize, embedParams.embeddingSize, embedParams.windowSize, embedParams.committee)
    }
    else if ("VE".equals(sampleName)) {
      new VEScorer(embedParams.dictionarySize, embedParams.embeddingSize, embedParams.windowSize, embedParams.committee)
    }
    else if ("LM".equals(sampleName)) {
      new LanguageModelScorer(embedParams.dictionarySize, embedParams.embeddingSize)
    }

    else if ("Boltzmann".equals(sampleName)) {
      new BoltzmannScorer(embedParams.secondDictionarySize, embedParams.embeddingWindowSize, embedParams.hiddenSize)
    }
    else if ("Least".equals(sampleName)) {
      new LeastSquares(embedParams.maxInitSamples, embedParams.secondDictionarySize, embedParams.embeddingSize)
    }
    else if ("Hopfield".equals(sampleName)) {
      new HopfieldScorer(embedParams.embeddingSize, embedParams.embeddingWindowSize)
    }
    else if ("Mahalonabis".equals(sampleName)) {
      new MahalonabisScore(embedParams.dictionarySize, embedParams.embeddingSize)
    }
    else if ("VotedDivergence".equals(sampleName)) {
      new VotedDivergence(embedParams.secondDictionarySize, embedParams.secondDictionarySize, embedParams.windowSize, embedParams.committee)
    }
    else if ("Entropy".equals(sampleName)) {
      new BinaryEntropyScorer(embedParams.dictionarySize, embedParams.secondDictionarySize)
    }
    else if (sampleName.startsWith("Cosine")) {
      new CosineScorer(embedParams.dictionarySize, embedParams.embeddingSize)
    }
    else if (sampleName.startsWith("VocabSelect")) {
      new VoabSelect(embedParams.dictionarySize, embedParams.secondDictionarySize)
    }
    else if (sampleName.startsWith("Euclidean")) {
      new EuclideanScorer(embedParams.secondDictionarySize, embedParams.secondDictionarySize)
    }
    else if (sampleName.startsWith("Least")) {
      new LeastSquares(50, embedParams.dictionarySize, embedParams.embeddingSize)
    }
    else {
      null
    }

  }

  def createExtractor(name: String): Extractor = {
    if (name.equals("word")) new WordExtractor(embedParams.dictionarySize)
    else if (name.equals("ngram")) new NgramExtractor(embedParams.dictionarySize, embedParams.tokenLength)
    else if (name.equals("feature")) new FeatureExtractor(embedParams.dictionarySize, embedParams.secondDictionarySize, embedParams.tokenLength, embedParams.freqcutoff)
    else if (name.equals("hybrid")) new MultiExtractor(Array(new WordExtractor(embedParams.dictionarySize), new NgramExtractor(embedParams.dictionarySize, embedParams.tokenLength)))
    else if (name.equals("readability")) new ReadabilityExtractor()
    else null
  }





  def createMainDataset(sampleParams: SampleParams, extractor: Extractor, dictionaryFilename: String, datasetFilename: String): Iterator[TextInstance] = {
    val f = new File(datasetFilename)
    if (f.exists()) {
      Source.fromFile(f).getLines().filter(sentence => sentence.length > sampleParams.minSentenceLength)
        .map(sentence => extractor.itemize(new TextInstance(sentence)))
    }
    else {
      val pw = new PrintWriter(f)
      val evalDictionary = Source.fromFile(dictionaryFilename).getLines().toArray
      val mainSentences = Source.fromFile(sampleParams.sentenceFilename).getLines().toArray

      evalDictionary.foreach(word => mainSentences.par
        .filter(text => text.contains(word)).take(sampleParams.maxWordSamples).toArray
        .foreach(item => pw.println(item)))
      pw.close()

      createMainDataset(sampleParams, extractor, dictionaryFilename, datasetFilename)
    }
  }

  def createStatistics(dataset: Array[TextInstance], params: SampleParams): Unit = {
    val averageLength = dataset.map(instance => instance.text.length.toDouble).sum / dataset.length
    val totalTokenSize = dataset.map(instance => instance.featureSequence.map(_.length).sum).sum
    val averageTokenSize = totalTokenSize / dataset.length
    val totalDistinctSize = dataset.map(instance => instance.featureSequence.flatMap(items => items.distinct)).toSet.size.toDouble

    val pw = new PrintWriter(new FileOutputStream("resources/results/statistics/" + params.scorerName + ".stat", true))

    pw.println("=========================================================")
    pw.println("Method: " + params.scorerName)
    pw.println("Select size: " + params.maxSelectSize)
    pw.println("Average char length: " + averageLength)
    pw.println("Total token size: " + totalTokenSize)
    pw.println("Average token size: " + averageTokenSize)
    pw.println("Total distinct tokens: " + totalDistinctSize)
    pw.println("=========================================================")
    pw.close()
  }

  def createSamples(sampleParams: SampleParams): Unit = {

    println("Task: " + sampleParams.taskName)
    println("Extractor: " + sampleParams.extractorName)
    println("Sampling: " + sampleParams.scorerName)
    println("Adapter: " + sampleParams.adapterName)
    println("Selection size: " + sampleParams.maxSelectSize)

    val samplesFilename = sampleParams.sampledDataset()

    if (!new File(samplesFilename).exists()) {

      val extractor = createExtractor(sampleParams.extractorName)
      if (extractor.exists()) {
        extractor.load()
      }
      else {
        extractor.parbuild(createMainDataset(sampleParams, extractor, sampleParams.dictionaryFile(), sampleParams.mainDataset()))
          .save()
      }

      println("Filtering dataset....")
      val dataset = createMainDataset(sampleParams, extractor, sampleParams.dictionaryFile(), sampleParams.mainDataset()).map(textInstance => extractor.process(textInstance, 0)).filter(textInstance => {
        textInstance.features.nonEmpty
      })

      println("Initializing adapter")
      var selected = dataset.take(sampleParams.maxInitSamples)
        .toArray

      val scorer = createCriterias(sampleParams.scorerName)
      val adapter = createAdapter(sampleParams.adapterName, scorer, sampleParams.kmoving, sampleParams.kselectSize, sampleParams.maxSelectSize, sampleParams.threshold)
        .init(selected)

      val breaking = Breaks

      breaking.breakable {
        dataset.sliding(sampleParams.batchSize, sampleParams.batchSize).foreach(sentences => {
          println(s"Computing for ${sampleParams.scorerName}")
          //println(s"Computing for ${sampleParams.maxSelectSize}")
          adapter.status()
          if (adapter.isStop()) breaking.break()
          else {
            selected = selected ++ adapter.filter(sentences.toArray)
          }
        })
      }

      println("Create statistics")
      createStatistics(selected, sampleParams);

      new PrintWriter(samplesFilename) {
        selected.foreach(line => println(line.text))
      }.close()

      new PrintWriter("resources/results/efficiency/" + "sampling-" + sampleParams.experimentKey() + ".txt") {
        println("Extractor: " + sampleParams.extractorName)
        println("Scorer: " + sampleParams.scorerName)
        println("Total Items: " + sampleParams.maxSelectSize)
        println("AVG time: " + adapter.totalTime / adapter.count)
      }.close()
    }
    else {
      println("Found file:" + samplesFilename)
    }
  }

  def createSamples(): Unit = {

    taskNames.foreach(taskName=> {
      tokenization.foreach(extractorName => {
        selectionSizes.foreach(selectionSize => {
          val parCollection = samplingNames.par
          parCollection.tasksupport = new ForkJoinTaskSupport(new concurrent.ForkJoinPool(embedParams.nthreads))
          parCollection.foreach(scorerName => {

            adapterNames.foreach(adapterName => {
              if (adapterName.equals("avg")) {
                val copyParams = embedParams.copy()
                copyParams.taskName = taskName
                copyParams.adapterName = adapterName
                copyParams.maxSelectSize = selectionSize
                copyParams.scorerName = scorerName
                copyParams.extractorName = extractorName
                createSamples(copyParams)
              }
            })
          })
        })
      })
    })
  }

  def createMajority(): Unit = {
    taskNames.foreach(taskName=>{
      tokenization.foreach(extractorName => {
        selectionSizes.foreach(selectionSize => {
          embedParams.taskName = taskName
          embedParams.adapterName = "majority"
          embedParams.maxSelectSize = selectionSize
          embedParams.scorerName = "ALL"
          embedParams.extractorName = extractorName
          createSamples(embedParams)
        })
      })
    })

  }

  def main(args: Array[String]): Unit = {
    System.setProperty("org.bytedeco.openblas.load", "mkl")
    createSamples()
    createMajority()

  }
}


