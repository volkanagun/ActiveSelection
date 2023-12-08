package utils


class Params {

  var epocs:Int = 5
  var batchSize:Int = 1

  var lrate = 0.001
  var hiddenLength:Int = 200
  var nheads:Int = 4
  var embeddingLength:Int = 100
  var modelWindowLength:Int = 20

  var dictionarySize:Int = 200000
  var freqCutoff:Int = 0


  var evalWindowLength: Int = 20
  var sentimentSize:Int = 3
  var nerSize:Int = 10
  var posSize:Int = 12

  var topicSize:Int = 7
  var categorySize:Int = 7


  var maxSentences = 10000000
  var maxSelectSize = 10000000
  var selectionSize = 250000


  var selectionMethod = "Hopfield"
  var adapterName = "avg"

  var committee = 10
  var tokenLength = 5
  var clusterSize = 20
  var knn = 7
  var windowLength = 10

  var minSentenceLength = 0
  var maxSentenceLength = 500


  var embeddingModel = "CBOW"


  var nthreads = 4

  val modelFolder = "resources/models/"
  val textFolder = "resources/text/"
  val intrinsicTextFolder = "resources/text/intrinsic/"

  val evaluationFolder = "resources/evaluation/"
  val dictionaryFolder = "resources/dictionary/"
  val resultFolder = "resources/results/"


  def copy():Params={
    val params = new Params()
    params.embeddingModel = embeddingModel
    params.selectionMethod = selectionMethod
    params.adapterName = adapterName

    params.epocs = epocs
    params.lrate = lrate
    params.hiddenLength = hiddenLength
    params.nheads=nheads
    params.embeddingLength = embeddingLength
    params.modelWindowLength = modelWindowLength
    params.dictionarySize = dictionarySize
    params.freqCutoff = freqCutoff
    params.evalWindowLength = evalWindowLength
    params.sentimentSize = sentimentSize
    params.topicSize = topicSize
    params.categorySize = categorySize
    params.maxSentences = maxSentences
    params.maxSelectSize = maxSelectSize
    params.selectionSize = selectionSize
    params.committee = committee
    params.tokenLength = tokenLength
    params.clusterSize = clusterSize
    params.knn = knn
    params.windowLength = windowLength
    params.minSentenceLength = minSentenceLength
    params.maxSentenceLength = maxSentenceLength
    params.nthreads = nthreads
    params
  }

  def modelName(name:String):this.type ={
    embeddingModel = name
    this
  }

 /* def createModel(name:String):EmbeddingModel={
    if("cbow".equals(name)) new CBOWModel(this)
    else if("skip".equals(name)) new SkipGramModel(this)
    else if("lstm".equals(name)) new SelfAttentionLSTM(this)
    else null
  }*/

  def corpusFilename():String={
    textFolder + corpusID() + ".txt"
  }
  def resultFilename(modelling:String):String={
    val fname = resultFolder + modelling
    fname + "/" + modelID() + ".xml"
  }
  def modelFilename():String={
    modelFolder + modelID() + ".bin"
  }
  def dictionaryFilename():String={
    dictionaryFolder + modelID() + ".bin"
  }

  def modelID():Int = {
    Array[Int](lrate.hashCode(), embeddingModel.hashCode, hiddenLength, nheads, embeddingLength, modelWindowLength, dictionarySize).foldRight(7){case(a, main)=> a + 7 * main}
  }

  def corpusID():Int={
    Array[Int](selectionSize, selectionMethod.hashCode, adapterName.hashCode).foldRight(7){case(a, main)=> a + 7 * main}
  }

  def tag(name:String, value:String):String={
    s"<PARAM LABEL=\"${name}\" VALUE=\"${value}\"/>\n"
  }

  def toShortXML():String={
    "<MODEL_PARAMETERS>\n" +
      tag("MODEL", embeddingModel) +
      tag("EPOCS", epocs.toString) +
      tag("LEARNING_RATE", lrate.toString) +
      tag("HIDDEN_LENGTH", hiddenLength.toString)+
      tag("EMBEDDING_LENGTH", embeddingLength.toString)+
      tag("WINDOW_LENGTH", modelWindowLength.toString)+
      tag("HEADS", nheads.toString)+
      tag("DICTIONARY_SIZE", dictionarySize.toString)+
      tag("SELECTION_SIZE", selectionSize.toString) +
    "</MODEL_PARAMETERS>\n" +
    "<SELECTION_PARAMETERS>\n" +
      tag("SELECTION_METHOD", selectionMethod) +
      tag("ADAPTER_NAME", adapterName) +
      tag("MAX_SENTENCES", maxSentences.toString) +
      tag("MAX_SELECT_SIZE", maxSelectSize.toString) +
      tag("SELECTION_SIZE", selectionSize.toString) +
      tag("COMITTEE", committee.toString) +
      tag("TOKEN_LENGTH", tokenLength.toString) +
      tag("CLUSTER_SIZE", clusterSize.toString) +
      tag("KNN", knn.toString) +
      tag("WINDOW_LENGTH", windowLength.toString) +
      tag("MIN_SENTENCE_LENGTH", minSentenceLength.toString) +
      tag("MAX_SENTENCE_LENGTH", maxSentenceLength.toString) +
      tag("NTHREADS", nthreads.toString) +
      "</SELECTION_PARAMETERS>\n"
  }
}
