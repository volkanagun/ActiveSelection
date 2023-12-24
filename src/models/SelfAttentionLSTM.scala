package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.parallelism.ParallelWrapper
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sampling.experiments.SampleParams
import utils.Tokenizer

import java.io._
import scala.io.Source

class SelfAttentionLSTM(params: SampleParams, tokenizer: Tokenizer, isEvaluation: Boolean = false) extends EmbeddingModel(params, tokenizer, isEvaluation) {


  def onehot(indices: Array[Int]): INDArray = {

    val ndarray = Nd4j.zeros(1, params.embeddingDictionarySize, params.embeddingWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })

    for (i <- indices.length until params.embeddingWindowLength) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }
    ndarray
  }


  def index(indices: Array[Int]): INDArray = {
    val ndarray = Nd4j.zeros(1, params.embeddingWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._2)), pair._1.toFloat)
    })

    for (i <- indices.length until params.embeddingWindowLength) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }
    ndarray
  }

  def index(indices: Array[Int], windowSize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, windowSize)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._2)), pair._1.toFloat)
    })

    for (i <- indices.length until windowSize) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }
    ndarray
  }


  def onehot(indices: Array[Int], categorySize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize, params.embeddingWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })

    for (i <- indices.length until params.embeddingWindowLength) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }

    ndarray
  }

  def onehot(indices: Array[Int], categorySize: Int, windowSize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize, windowSize)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })

    for (i <- indices.length until windowSize) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }

    ndarray
  }

  def onehot(indice: Int, categorySize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize)
    ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice)), 1f)
    ndarray
  }

  def vectorize(array: Array[Array[Float]]): INDArray = {
    val sz1 = params.embeddingWindowLength
    val sz2 = params.embeddingLength
    val ndarray = Nd4j.zeros(1, sz2, sz1)
    array.zipWithIndex.foreach { case (sub, windex) => {
      val embedding = Nd4j.create(sub)
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(windex)), embedding)
    }
    }

    ndarray
  }

  def vectorize(sentence: String): Array[INDArray] = {
    tokenize(sentence).sliding(params.embeddingWindowLength, params.embeddingWindowLength)
      .map(ngrams => onehot(ngrams.map(update(_))))
      .toArray
  }

  def vectorizeIndex(sentence: Array[String]): Array[INDArray] = {
    sentence.sliding(params.embeddingWindowLength, 1)
      .map(ngrams => index(ngrams.take(params.embeddingWindowLength - 1).map(update(_))))
      .toArray
  }

  def vectorizeOneHotLast(sentence: Array[String]): Array[INDArray] = {
    sentence.sliding(params.embeddingWindowLength, 1)
      .map(ngrams => onehot(update(ngrams.last), params.embeddingDictionarySize))
      .toArray
  }

  def maskInput(sentenceVector: Array[INDArray]): Array[INDArray] = {
    sentenceVector.map(_ => {
      val mask = Nd4j.ones(1, params.embeddingWindowLength)
      mask.put(Array(NDArrayIndex.point(params.embeddingWindowLength - 1)), 0f)
      mask
    })
  }

  def maskOutput(sentenceVector: Array[INDArray]): Array[INDArray] = {
    sentenceVector.map(_ => {
      val mask = Nd4j.zeros(1, 1)
      mask.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0)), 1f)
      mask
    })
  }

  def iterator(filename: String): MultiDataSetIterator = {

    new MultiDataSetIterator {

      var lines = Source.fromFile(filename).getLines()


      override def next(i: Int): MultiDataSet = {
        var cnt = 0
        var trainStack = Array[INDArray]()
        var trainOutputStack = Array[INDArray]()
        //var maskInputStack = Array[INDArray]()
        //var maskOutputStack = Array[INDArray]()


        while (cnt < params.batchSize && hasNext) {
          val sentence = lines.next()
          val frequentNgrams = tokenize(sentence)

          val sentenceVector = vectorizeIndex(frequentNgrams)
          val lastWordVector = vectorizeOneHotLast(frequentNgrams)

          trainStack = trainStack ++ sentenceVector
          trainOutputStack = trainOutputStack ++ lastWordVector
          //maskInputStack = maskInputStack ++ maskInput(sentenceVector)
          //maskOutputStack = maskOutputStack ++ maskOutput(sentenceVector)
          cnt += sentenceVector.length

        }

        //val maskingInput = Nd4j.vstack(maskInputStack: _*)
        //val maskingOutput = Nd4j.vstack(maskOutputStack: _*)
        val trainVector = Nd4j.vstack(trainStack: _*)
        val trainOutputVector = Nd4j.vstack(trainOutputStack: _*)
        //new org.nd4j.linalg.dataset.MultiDataSet(trainVector, trainOutputVector, maskingInput, maskingOutput)
        new org.nd4j.linalg.dataset.MultiDataSet(trainVector, trainOutputVector)
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = false

      override def reset(): Unit = {
        lines = Source.fromFile(filename).getLines()

      }

      override def hasNext: Boolean = lines.hasNext

      override def next(): MultiDataSet = next(0)
    }
  }


  def getWrapper(model: ComputationGraph): ParallelWrapper = {
    new ParallelWrapper.Builder(model)
      .prefetchBuffer(24)
      .workers(2).averagingFrequency(3).reportScoreAfterAveraging(true)
      .build
  }

  override def train(filename: String): EmbeddingModel = {

    var i = 0
    val modelFile = new File(params.modelFilename())
    val size = Source.fromFile(filename).getLines().size
    val epocs = (if(isEvaluation) params.evalEpocs else params.epocs)

    load()

    if (!new File(params.embeddingsFilename()).exists() || isEvaluation) {
      println("Training started...")
      computationGraph = model()
      val statsStorage = new InMemoryStatsStorage()
      val uiServer = UIServer.getInstance()
      uiServer.attach(statsStorage)

      computationGraph.addListeners(new StatsListener(statsStorage, 1))

      val multiDataSetIterator = iterator(filename)

      val start = System.currentTimeMillis()

      sampleCount = 0
      //val wrapper = getWrapper(computationGraph)

      while (i < epocs) {

        println()
        println("Epoc : " + i)

        computationGraph.fit(multiDataSetIterator)
        //wrapper.fit(multiDataSetIterator)
        multiDataSetIterator.reset()

        i = i + 1
        sampleCount += size

        //System.gc()
      }

      val passedTime = System.currentTimeMillis() - start
      avgTime = passedTime / (sampleCount)

      println("Saving model...")
      computationGraph.save(modelFile)
      if (!isEvaluation) saveEmbeddings(computationGraph)

      ModelSerializer.writeModel(computationGraph, modelFile, false)
      uiServer.stop()

      System.gc()
    }
    else if (new File(params.embeddingsFilename()).exists()) {
      println("Embedding filename exists: " + params.embeddingsFilename())
    }


    this

  }


  def saveEmbeddings(net: ComputationGraph): EmbeddingModel = {
    val weights = net.getVertex("embedding").paramTable(false).get("W")
    //val weightsMatrix = weights.toFloatMatrix
    val objectOutput = new ObjectOutputStream(new FileOutputStream(params.embeddingsFilename()))
    objectOutput.writeInt(dictionaryIndex.size)
    dictionaryIndex.foreach { case (ngram, id) => {
      val embeddingVector = weights.get(NDArrayIndex.point(id)).toFloatVector
      objectOutput.writeObject(ngram)
      objectOutput.writeObject(embeddingVector)
      dictionary = dictionary.updated(ngram, embeddingVector)
    }
    }

    objectOutput.close()
    this
  }

  override def load(): SelfAttentionLSTM.this.type = {
    if (new File(params.embeddingsFilename()).exists()) {
      println("Loading embedding dictionary")
      val objectInput = new ObjectInputStream(new FileInputStream(params.embeddingsFilename()))
      val size = objectInput.readInt()
      Range(0, size).foreach { index => {
        val ngram = objectInput.readObject().asInstanceOf[String]
        val vector = objectInput.readObject().asInstanceOf[Array[Float]]
        dictionary = dictionary.updated(ngram, vector)
        dictionaryIndex = dictionaryIndex.updated(ngram, index)
      }}

      objectInput.close()
    }
    this

  }

  def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .layer("embedding", new EmbeddingSequenceLayer.Builder()
        .inputLength(params.embeddingWindowLength - 1)
        .nIn(params.embeddingDictionarySize)
        .nOut(params.embeddingLength).build(), "input")
      .layer("input-lstm", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.TANH).build, "embedding")
      .layer("attention", new SelfAttentionLayer.Builder().nOut(params.embeddingHiddenLength).nHeads(params.nheads).projectInput(true).build(), "input-lstm")
      .layer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
      .layer("dense_base", new DenseLayer.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingHiddenLength).activation(Activation.SIGMOID).build(), "pooling")
      .layer("dense", new DenseLayer.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingHiddenLength).activation(Activation.SIGMOID).build(), "dense_base")
      .layer("output", new OutputLayer.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingDictionarySize).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "dense")
      .setInputTypes(InputType.recurrent(params.embeddingDictionarySize))
      .build()

    new ComputationGraph(conf)
  }

  override def save(): EmbeddingModel = {
    saveEmbeddings(computationGraph)
    this
  }

  override def evaluate(model: EmbeddingModel): EvalScore = EvalScore(0, 0)

  override def setDictionary(set: Set[String], model: EmbeddingModel): SelfAttentionLSTM.this.type = this

  override def setWords(set: Set[String]): SelfAttentionLSTM.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): SelfAttentionLSTM.this.type = this

  override def count(): Int = 0

  override def universe(): Set[String] = Set()

  override def getClassifier(): String = "self-attention"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
