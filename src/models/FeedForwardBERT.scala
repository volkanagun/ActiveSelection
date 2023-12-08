package models

import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingSequenceLayer, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
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
import scala.io.Source
import scala.util.Random

class FeedForwardBERT(params: SampleParams, tokenizer: Tokenizer) extends SelfAttentionLSTM(params, tokenizer) {
  //use BERT
  var dummy = "dummy"
  //var inputDictionary = Map[String, Int](dummy -> 0)
  var windowSize = 5 //params.embeddingWindowLength
  var dictionarySize = 100000 //params.embeddingWindowLength



  case class Pairs(input: Array[String], output: Array[String], inputMask: Array[Float], outputMask: Array[Float])

  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = this

  override def setWords(set: Set[String]): this.type = this

  def pair(tokens: Array[String]): Array[Pairs] = {
    var inputOutput = Array[Pairs]()
    //println("Processing line")

    tokens.sliding(windowSize, 1).foreach(output => {

      for (i <- 0 until params.embeddingRandomMask) {
        val index = Random.nextInt(windowSize)
        val maskInput = Array.fill[Float](output.length)(1f)
        val maskOutput = Array.fill[Float](output.length)(1f)

        val input = Array.copyOf(output, output.length)
        maskInput(index) = 0
        input(index) = dummy
        inputOutput = inputOutput :+ Pairs(input, output, maskInput, maskOutput)
      }
    })

    inputOutput
  }


  override def iterator(filename: String): MultiDataSetIterator = {


    new MultiDataSetIterator {

      var lines = Source.fromFile(filename).getLines()
      var stack = mutable.Stack[Pairs]()

      override def next(i: Int): MultiDataSet = {
        var inputStack = Array[INDArray]()
        var inputMaskStack = Array[INDArray]()
        var outputStack = Array[INDArray]()
        var outputMaskStack = Array[INDArray]()
        var i = 0;
        while (i < params.batchSize && lines.hasNext) {

          if (stack.isEmpty) {
            val input = lines.next()
            val tokens = tokenize(input)
            val inputOutput = pair(tokens)
            stack = stack ++ inputOutput
          }

          val Pairs(input, output, maskInput, maskOutput) = stack.pop()

          val inputArray = input.map(ngram => {
            update(ngram)
          })

          val outputArray = output.map(ngram => {
            update(ngram)
          })

          val inputIndex = index(inputArray, windowSize)
          val outputIndex = onehot(outputArray, dictionarySize, windowSize)

          val inputMaskArray = Nd4j.create(maskInput)
          val outputMaskArray = Nd4j.create(maskOutput)

          inputStack :+= inputIndex
          outputStack :+= outputIndex
          inputMaskStack :+= inputMaskArray
          outputMaskStack :+= outputMaskArray
          i = i + 1

        }

        val vInput = Nd4j.vstack(inputStack: _*)
        val vOutput = Nd4j.vstack(outputStack: _*)

        val vInputMask = Nd4j.vstack(inputMaskStack: _*)
        val vOutputMask = Nd4j.vstack(outputMaskStack: _*)

        new MultiDataSet(vInput, vOutput, vInputMask, vOutputMask)
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

  override def model(): ComputationGraph = {

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
      .addLayer("dense", new DenseLayer.Builder().nIn(params.embeddingLength * windowSize).nOut(params.hiddenSize).build(), "embedding")
      //.addLayer("dense2", new DenseLayer.Builder().nIn(params.hiddenSize).nOut(params.embeddingLength * windowSize).build(), "dense1")
      //.addVertex("preoutput", new PreprocessorVertex(new FeedForwardToRnnPreProcessor(RNNFormat.NWC)), "dense2")
      .addLayer("output",
        new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .nOut(dictionarySize)
          .dataFormat(RNNFormat.NCW)
          .build(), "dense")
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(dictionarySize))
      .build()

    val graph = new ComputationGraph(conf)
    graph.init()
    graph
  }

}
