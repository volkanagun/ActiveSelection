package evaluation

import models.{EmbeddingModel, SelfAttentionLSTM}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.evaluation.classification.Evaluation
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

import scala.collection.mutable
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source

abstract class ExtrinsicSelf(params: SampleParams, tokenizer: Tokenizer, forceTrain: Boolean = false) extends SelfAttentionLSTM(params, tokenizer, forceTrain) {
  //use ELMO
  var inputDictionary = Map[String, Int]("dummy" -> 0)
  var maxWindowLength = params.evalWindowSize

  def getTraining(): String

  def getTesing(): String

  def loadSamples(filename: String): Iterator[(String, String)]

  def labels(): Array[String]

  override def filter(group: Array[String]): Boolean = true

  override def count(): Int = 1

  override def universe(): Set[String] = {
    Source.fromFile(getTraining()).getLines().flatMap(sentence => {
      val tokens = sentence.split("\\s+")
        .map(token => token.split("\\/").head)
      tokens
    }).toSet
  }


  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = this

  override def setWords(set: Set[String]): this.type = this

  def init(): this.type = {
    /*val lengths = loadSamples(getTraining()).toArray.par.map(pair=> tokenize(pair._1).length)
    maxWindowLength = lengths.max*/
    this
  }

  def evaluate(): EvalScore = {
    //Train
    val trainingFilename = getTraining()
    val testingFilename = getTesing()

    train(trainingFilename)
    val evaluation: Evaluation = computationGraph.evaluate(iterator(testingFilename))

    //Test TP Rates
    EvalScore(evaluation.accuracy(), evaluation.f1())
  }


  override def evaluate(model: EmbeddingModel): EvalScore = {
    evaluate()
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport().incrementTestPair()
    val classifier = getClassifier()
    ier.incrementQueryCount(classifier, 1d)

    val value = evaluate(model)

    ier.incrementTruePositives(value.tp)
    ier.incrementScoreMap(classifier, value.tp)

    ier.incrementSimilarity(value.similarity)
    ier.incrementSimilarityMap(classifier, value.similarity)


    ier.printProgress(classifier)
    ier
  }


  def retrieve(tokens:Array[String]):Array[Int]={
    var array = Array[Int]()
    val length = tokens.length
    tokens.foreach(ngram=> array:+= retrieve(ngram))
    for(i<-length until maxWindowLength){
      array :+= retrieve("dummy")
    }
    array
  }


  override def iterator(filename: String): MultiDataSetIterator = {
    new MultiDataSetIterator {
      val dataset = loadSamples(filename).toArray
      var lines = dataset.iterator
      val maxSize = dataset.length
      val maxLength = 100;
      val stepSize = maxSize / maxLength
      var crrCount = 0;
      var stack = mutable.Stack[(String, String)]()

      override def next(i: Int): MultiDataSet = {
        var inputStack = Array[INDArray]()
        var outputStack = Array[INDArray]()
        var inputMaskStack = Array[INDArray]()
        var outputMaskStack = Array[INDArray]()
        var i = 0;
        while (i < params.evalBatchSize && lines.hasNext) {

          crrCount+=1

          val (input, output) = lines.next()

          val inputArray = retrieve(tokenize(input).take(maxWindowLength))
          val inputMask = mask(inputArray.length, maxWindowLength)
          val inputIndex = index(inputArray, maxWindowLength)
          val outputArray = onehot(labels().indexOf(output), labels.length)
          val outputMask = mask(1, 1)
          inputStack :+= inputIndex
          inputMaskStack :+= inputMask
          outputStack :+= outputArray
          outputMaskStack :+= outputMask
          i = i + 1

        }

        progress("PROGRESS", maxLength, crrCount / stepSize)

        val stackInput = Nd4j.vstack(inputStack: _*)
        val stackMaskInput = Nd4j.vstack(inputMaskStack: _*)
        val stackOutput = Nd4j.vstack(outputStack: _*)
        val stackMaskOutput = Nd4j.vstack(outputMaskStack: _*)

        new MultiDataSet(stackInput, stackOutput, stackMaskInput, stackMaskOutput)
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = true

      override def reset(): Unit = {
        lines = loadSamples(filename)
        crrCount = 0
      }

      override def hasNext: Boolean = lines.hasNext

      override def next(): MultiDataSet = next(0)
    }
  }

  def updateWeights(graph: ComputationGraph): ComputationGraph = {


    graph.init()

    if (params.evalUseEmbeddings) {
      val vertex = graph.getVertex("embedding")
      val weight = vertex.paramTable(false).get("W")
      dictionaryIndex.foreach { case (ngram, index) => {
        val array = dictionary(ngram)
        weight.put(Array(NDArrayIndex.point(index), NDArrayIndex.all()), Nd4j.create(array))
      }
      }

      //Use exiting weights and also new weights together
      //They can not be updated as well.
      vertex.setLayerAsFrozen()
    }

    graph
  }

  override def model(): ComputationGraph = {
    load()

    val dictionarySize = dictionaryIndex.size

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.evalRate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .layer("embedding", new EmbeddingSequenceLayer.Builder()
        .inputLength(params.embeddingWindowLength)
        .nIn(dictionarySize)
        .nOut(params.embeddingLength).build(), "input")
      .layer("input-lstm", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.TANH).build, "embedding")
      .layer("attention", new SelfAttentionLayer.Builder().nOut(params.embeddingHiddenLength).nHeads(params.nheads).projectInput(true).build(), "input-lstm")
      .layer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
      //.layer("dense_base", new DenseLayer.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingHiddenLength).activation(Activation.SIGMOID).build(), "pooling")
      //.layer("dense", new DenseLayer.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingHiddenLength).activation(Activation.SIGMOID).build(), "dense_base")
      .layer("output", new OutputLayer.Builder().nIn(params.embeddingHiddenLength).nOut(labels().length).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "pooling")
      .setInputTypes(InputType.recurrent(dictionarySize))
      .build()

    val graph = new ComputationGraph(conf)
    updateWeights(graph)
  }

  /*override def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .updater(new Adam.Builder().learningRate(params.evalRate).build())
      .dropOut(0.5)
      .graphBuilder()
      .allowDisconnected(true)
      .addInputs("left", "right")
      .addVertex("stack", new org.deeplearning4j.nn.conf.graph.StackVertex(), "left", "right")
      .addLayer("embedding", new EmbeddingSequenceLayer.Builder().inputLength(maxWindowLength)
        .nIn(params.evalDictionarySize).nOut(params.embeddingLength).build(),
        "stack")
      .addVertex("leftemb", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "embedding")
      .addVertex("rightemb", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "embedding")
      //can use any label for this
      .addLayer("leftout", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.RELU)
        .build(), "leftemb")
      .addLayer("rightout", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.RELU)
        .build(), "rightemb")
      .addVertex("merge", new MergeVertex(), "leftout", "rightout")
      .addLayer("output-lstm", new LastTimeStep(new LSTM.Builder().nIn(params.embeddingHiddenLength*2).nOut(params.embeddingHiddenLength)
        .activation(Activation.RELU)
        .build()), "merge")
      .addLayer("output",
        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .nOut(labels().length)
          .build(), "output-lstm")
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(params.evalDictionarySize),
        InputType.recurrent(params.evalDictionarySize))
      .build()

    val graph = new ComputationGraph(conf)
    graph.init()
    updateWeights(graph)

  }*/


}
