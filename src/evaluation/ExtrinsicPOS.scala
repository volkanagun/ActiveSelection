package evaluation

import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, WorkspaceMode}
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

import scala.io.Source
import scala.util.Random

class ExtrinsicPOS(params: SampleParams, tokenizer: Tokenizer) extends ExtrinsicELMO(params, tokenizer, true) {

  var categories: Array[String] = null


  override def getClassifier(): String = "pos"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/pos/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/pos/test.txt"
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {

    val rnd = new Random()
    val array = Source.fromFile(filename).getLines().toSeq
    rnd.shuffle(array).iterator.map(line => {
      val input = line.split("\t").head
      (input.toLowerCase(locale), input.toLowerCase(locale))
    })
  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if (categories == null) {
      println("Finding category labels")
      categories = Source.fromFile(getTraining()).getLines().map(line=> line.toLowerCase(locale))
        .flatMap(item => item.split("\\s").map(_.split("/").last).distinct)
        .toSet.toArray
      categories ="NONE" +: categories
      categories
    }
    else {
      categories
    }
  }

  override def iterator(filename: String): MultiDataSetIterator = {
    new MultiDataSetIterator {

      var samples = loadSamples(filename)
      val categorySize = labels().length


      override def next(i: Int): MultiDataSet = {
        var inputLeftStack = Array[INDArray]()
        var inputRightStack = Array[INDArray]()
        var outputStack = Array[INDArray]()
        var i = 0;
        while (i < params.evalBatchSize && samples.hasNext) {
          val (input, output) = samples.next()
          val tokens = input.split("\\s+")
          val inputOutput = tokens.map(word => word.split("/")).take(params.embeddingWindowLength)

          val inputLeft = inputOutput.map(_.head)

          val inputLeftArray = inputLeft.map(token => {
            val embeddings = forward(token)
            embeddings
          })

          val inputLeftIndex = vectorize(inputLeftArray)
          val inputRightIndex = vectorize(inputLeftArray.reverse)
          val outputElements = inputOutput.map(_.last)
          val outputIndices = outputElements
            .map(output => labels().indexOf(output))

          val outputArray = onehot(outputIndices, categorySize)


          inputLeftStack :+= inputLeftIndex
          inputRightStack :+= inputRightIndex
          outputStack :+= outputArray
          i = i + 1
        }

        val vLeft = Nd4j.vstack(inputLeftStack: _*)
        val vRight = Nd4j.vstack(inputRightStack: _*)
        val vOutput = Nd4j.vstack(outputStack: _*)


        new MultiDataSet(Array(vLeft, vRight), Array(vOutput))
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = false

      override def reset(): Unit = {
        samples = loadSamples(filename)
      }

      override def hasNext: Boolean = samples.hasNext

      override def next(): MultiDataSet = next(0)
    }
  }

  override def model(): ComputationGraph = {

    val categorySize = labels().length
    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.PREFER_FASTEST)
      .updater(new Adam.Builder().learningRate(params.lrate).build())
      .dropOut(0.2)
      .graphBuilder()
      .allowDisconnected(true)
      .addInputs("leftemb", "rightemb")
      .addLayer("leftout", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.RELU)
        .build(), "leftemb")
      .addLayer("rightout", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.RELU)
        .build(), "rightemb")
      .addVertex("merge", new MergeVertex(), "leftout", "rightout")
      .addLayer("output-lstm", new LSTM.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingHiddenLength)
        .activation(Activation.RELU)
        .build(), "merge")
      .addLayer("output",
        new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .nOut(categorySize)
          //.dataFormat(RNNFormat.NWC)
          .build(), "output-lstm")
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(params.embeddingLength),
        InputType.recurrent(params.embeddingLength))
      .build()

    conf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED)
    val graph = new ComputationGraph(conf)

    //updateWeights(graph)
    graph
  }
}
