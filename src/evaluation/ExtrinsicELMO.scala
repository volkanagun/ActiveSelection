package evaluation

import models.{EmbeddingModel, SelfAttentionLSTM}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
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

import scala.io.Source

abstract class ExtrinsicELMO(params: SampleParams, tokenizer: Tokenizer, forceTrain:Boolean = false) extends SelfAttentionLSTM(params, tokenizer, forceTrain)  {
  //use ELMO
  var inputDictionary = Map[String, Int]("dummy" -> 0)

  def getTraining():String
  def getTesing():String

  def loadSamples(filename: String): Iterator[(String, String)]

  def labels(): Array[String]

  override def filter(group: Array[String]): Boolean = true

  override def count(): Int = 1

  override def universe(): Set[String] = {
    Source.fromFile(getTraining()).getLines().flatMap(sentence=>{
     val tokens = sentence.split("\\s+")
       .map(token=> token.split("\\/").head)
     tokens
    }).toSet
  }


  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = this

  override def setWords(set: Set[String]): this.type = this

  def evaluate():EvalScore = {
    //Train
    val trainingFilename = getTraining()
    val testingFilename = getTesing()

    train(trainingFilename)
    val evaluation:Evaluation = computationGraph.evaluate(iterator(testingFilename))

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


  override def iterator(filename: String): MultiDataSetIterator = {
    new MultiDataSetIterator {

      var samples = loadSamples(filename)

      override def next(i: Int): MultiDataSet = {
        var inputLeftStack = Array[INDArray]()
        var inputRightStack = Array[INDArray]()
        var outputStack = Array[INDArray]()
        var i=0;
        while(i < params.evalBatchSize && samples.hasNext) {
          val (input, output) = samples.next()
          val tokens = tokenize(input)

          tokens.sliding(params.embeddingWindowLength, 1).foreach(inputLeft=>{
            val inputLeftArray = inputLeft.map(ngram => {
              update(ngram)
            })
            val inputLeftOneIndex = index(inputLeftArray)
            val inputRightOneIndex = index(inputLeftArray.reverse)
            val outputArray = onehot(labels().indexOf(output), labels.length)

            inputLeftStack :+= inputLeftOneIndex
            inputRightStack :+= inputRightOneIndex
            outputStack :+= outputArray
            i=i + 1
          })
        }

        val vLeft = Nd4j.vstack(inputLeftStack:_*)
        val vRight = Nd4j.vstack(inputRightStack:_*)
        val vOutput = Nd4j.vstack(outputStack:_*)

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

  def updateWeights(graph: ComputationGraph): ComputationGraph = {

    load()
    graph.init()

    if(params.evalUseEmbeddings) {
      val vertex = graph.getVertex("embedding")
      val weight = vertex.paramTable(false).get("W")
      dictionaryIndex.foreach { case (ngram, index) => {
        val array = dictionary(ngram)
        weight.put(Array(NDArrayIndex.point(index), NDArrayIndex.all()), Nd4j.create(array))
      }}

      //Use exiting weights and also new weights together
      //They can not be updated as well.
      vertex.setLayerAsFrozen()
    }

    graph
  }

  override def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .updater(new Adam.Builder().learningRate(params.lrate).build())
      .dropOut(0.5)
      .graphBuilder()
      .allowDisconnected(true)
      .addInputs("left", "right")
      .addVertex("stack", new org.deeplearning4j.nn.conf.graph.StackVertex(), "left", "right")
      .addLayer("embedding", new EmbeddingSequenceLayer.Builder().inputLength(params.embeddingWindowLength)
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
      .addLayer("output-lstm", new LastTimeStep(new LSTM.Builder().nIn(params.embeddingHiddenLength).nOut(params.embeddingHiddenLength)
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


  }

}
