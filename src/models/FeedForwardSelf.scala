package models

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sampling.experiments.SampleParams
import utils.Tokenizer

import scala.collection.mutable
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source

class FeedForwardSelf(params: SampleParams, tokenizer: Tokenizer) extends SelfAttentionLSTM(params, tokenizer) {
  //use BERT
  var dummy = "dummy"
  //var inputDictionary = Map[String, Int](dummy -> 0)
  val windowSize = params.embeddingWindowLength
  var maxSize = 0
  var stepSize = 50
  var maxLength:Int = 0
  var dictionarySet = Set[String]()
  var dictionarySize = 0

  init(params.embeddingDictionarySize)

  case class Pairs(input: Array[String], output: Array[String], inputMask: Array[Float], outputMask: Array[Float])

  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = this

  override def setWords(set: Set[String]): this.type = this

  def pair(tokens: Array[String]): Array[Pairs] = {
    var inputOutput = Array[Pairs]()

    //println("Processing line")
    val ntokens = tokens ++ Array.fill[String](Math.max(windowSize-tokens.length, 0))(dummy)

    ntokens.sliding(windowSize, 1).foreach(tokens => {
      for (i <- 0 until params.embeddingRandomMask) {
        val index =  windowSize - 1
        val maskInput = Array.fill[Float](windowSize)(1f)
        val maskOutput = Array.fill[Float](windowSize)(1f)
        val output = Array(tokens(index))
        val input = Array.copyOf(tokens, tokens.length)
        maskInput(index) = 0

        input(index) = dummy
        inputOutput = inputOutput :+ Pairs(input, output, maskInput, maskOutput)
      }
    })

    inputOutput
  }

  def init(size:Int):Unit= {

    //Compute dictionary size
    val dataset = (Source.fromFile(params.sampledDataset())
      .getLines().toArray).par
      .map(line => tokenize(line))
      .toArray

    println("Dataset size: " + dataset.length)

    maxSize = dataset.length
    maxLength = 100;
    stepSize = maxSize / maxLength

    val array = dataset.flatMap(tokens=> tokens)
      .groupBy(item=> item).map(pair=> (pair._1, pair._2.length))
      .toArray

    println("Total Dictionary size: " + array.size)

    dictionarySet = array.sortBy(_._2).reverse.take(size)
      .map(_._1).toSet + dummy

    dictionarySet.foreach(token=> dictionaryIndex = dictionaryIndex.updated(token, dictionaryIndex.size))

    dictionarySize = dictionarySet.size + 100
    println("Frequent Dictionary size: " + dictionarySize)

    params.embeddingDictionarySize = dictionarySize

  }

/*  override def progress(message: String, length: Int, crr:Int): Unit = {
    val incomplete = '░'
    val complete = '█'
    val builder = new StringBuilder()
    Range(0, length).map(_=> incomplete).foreach(builder.append)

    //System.out.println(message)
    for (i <- 0 until crr) {
      builder.replace(i, i + 1, String.valueOf(complete))
      val progressBar = "\r" + builder
      System.out.print(progressBar)
    }
  }*/


  override def iterator(filename: String): MultiDataSetIterator = {



    new MultiDataSetIterator {

      var lines = Source.fromFile(filename).getLines()
      var stack = mutable.Stack[Pairs]()

      var crrCount = 0

      override def next(i: Int): MultiDataSet = {
        var inputStack = Array[INDArray]()
        var inputMaskStack = Array[INDArray]()
        var outputStack = Array[INDArray]()
        var outputMaskStack = Array[INDArray]()
        var i = 0;

        while (i < params.batchSize && (lines.hasNext||stack.nonEmpty)) {

          if (stack.isEmpty && lines.hasNext) {
            val input = lines.next()
            val tokens = tokenize(input)
            val inputOutput = pair(tokens)
            stack = stack ++ inputOutput
            crrCount = crrCount + 1;
          }

          val Pairs(input, output, maskInput, maskOutput) = stack.pop()
          var inputArray = Array[Int]()

          input.foreach(ngram => {
            inputArray :+= retrieve(ngram)
          })

        /*  var outputArray = Array[Int]()
          output.foreach(ngram => {
            outputArray :+= update(ngram, dictionarySize)
          })*/

          val outputIndice = retrieve(output(0))
          val inputIndex = index(inputArray, windowSize)
          //val outputIndex = onehot(outputArray, dictionarySize, windowSize)
          val outputIndex = onehot(outputIndice, dictionarySize)

          val inputMaskArray = Nd4j.create(maskInput)
          val outputMaskArray = Nd4j.create(maskOutput)

          inputStack :+= inputIndex
          outputStack :+= outputIndex
          inputMaskStack :+= inputMaskArray
          outputMaskStack :+= outputMaskArray
          i = i + 1
        }

        progress("PROGRESS", maxLength, crrCount/stepSize)

        val vInput = Nd4j.vstack(inputStack: _*)
        val vOutput = Nd4j.vstack(outputStack: _*)

        //val vInputMask = Nd4j.vstack(inputMaskStack: _*)
        //val vOutputMask = Nd4j.vstack(outputMaskStack: _*)

        new MultiDataSet(vInput, vOutput/*, vInputMask, vOutputMask*/)
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = true

      override def reset(): Unit = {
        crrCount = 0;
        lines = Source.fromFile(filename).getLines()
      }

      override def hasNext: Boolean = lines.hasNext

      override def next(): MultiDataSet = next(0)
    }
  }

  /*override def model(): ComputationGraph = {



    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.PREFER_FASTEST)
      .updater(new Adam.Builder().learningRate(params.lrate).build())
      .graphBuilder()
      .allowDisconnected(true)
      .addInputs("input")
      .addLayer("embedding", new EmbeddingSequenceLayer.Builder().inputLength(windowSize)
        .nIn(dictionarySize).nOut(params.embeddingLength).build(),
        "input")
      //.addVertex("processor", new PreprocessorVertex(new RnnToFeedForwardPreProcessor(RNNFormat.NCW)),"embedding")
      .addLayer("dense1", new DenseLayer.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize/2).build(), "embedding")
      .addLayer("dense2", new DenseLayer.Builder().nIn(params.hiddenSize/2).nOut(params.hiddenSize).build(), "dense1")

      .addLayer("output",
        new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .nOut(dictionarySize)
          .dataFormat(RNNFormat.NCW)
          .build(), "dense2")
      /*.addLayer("output", new OutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
        .nOut(dictionarySize).nIn(params.windowSize).build(), "dense2")*/
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(dictionarySize))
      .build()

    //conf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    //conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED)

    val graph = new ComputationGraph(conf)
    graph.init()
    graph
  }*/

  override def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.PREFER_FASTEST)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
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
      .layer("output", new OutputLayer.Builder().nIn(params.embeddingHiddenLength).nOut(dictionarySize).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "pooling")
      .setInputTypes(InputType.recurrent(dictionarySize))
      .build()

    new ComputationGraph(conf)
  }
}
