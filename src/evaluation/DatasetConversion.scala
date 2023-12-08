package evaluation

import org.json4s.DefaultFormats
import utils.Tokenizer

import java.io.PrintWriter
import scala.io.Source
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse

import java.util.Locale

class DatasetConversion {

  val locale = new Locale("tr")

  def createSentimentVocabulary(): Unit = {
    val tokenizer = new Tokenizer()
    val sentimentFilename = "resources/evaluation/sentiment/train.txt"
    val filename = "resources/dictionary/sentiment.txt"
    val pw = new PrintWriter(filename)
    Source.fromFile(sentimentFilename).getLines()
      .filter(l=> l.contains("\t"))
      .take(10000)
      .map(line => {
        line.split("\\t").head
      }).flatMap(sentence => tokenizer.standardTokenizer(sentence)
        .filter(word => word.head.isLower)).toSet.toArray.foreach(word => {
        pw.println(word)
      })
    pw.close()
  }


  def createPOSVocabulary(): Unit = {

    val posFilename = "resources/evaluation/pos/train.txt"
    val filename = "resources/dictionary/pos.txt"
    val pw = new PrintWriter(filename)
    Source.fromFile(posFilename).getLines().flatMap(line => {
      line.split("[\t\\s]+").map(token => token.split("/").head)
        .filter(word => word.trim.nonEmpty)
        .filter(word => word.head.isLower && word.head.isLetter)
        .map(word => word.toLowerCase(locale))
    }).toSet.toArray.foreach(word => {
      pw.println(word)
    })
    pw.close()
  }

  def createNERVocabulary(): Unit = {

    val sentimentFilename = "resources/evaluation/ner/train.txt"
    val filename = "resources/dictionary/ner.txt"
    val pw = new PrintWriter(filename)
    implicit val formats = DefaultFormats
    val lines = Source.fromFile(sentimentFilename, "UTF-8").getLines()

    lines.flatMap(line => {
      line.split("[\\s\\t]+")
        .map(token => token.split("/").head)
        .filter(word => word.trim.nonEmpty)
        .filter(word => {
          word.head.isLower && word.head.isLetter
        })
        .map(_.toLowerCase(locale))
    }).toSet.toArray.foreach(word => {
      pw.println(word)
    })

    pw.close()
  }

  def convertNER(sentimentFilename: String, name: String): Unit = {

    val filename = "resources/evaluation/ner/" + name + ".txt"
    val pw = new PrintWriter(filename)
    implicit val formats = DefaultFormats
    val lines = Source.fromFile(sentimentFilename, "UTF-8").getLines()

    lines.foreach(line => {
      val json = parse(line)
      val tokens = (json \\ "tokens").children.map(_.extract[String])
      val tags = (json \\ "tags").children.map(_.extract[String])
      var sentence = ""
      tokens.zip(tags).foreach(pair => {
        sentence += pair._1 + "/" + pair._2 + " "
      })
      pw.println(sentence.trim)
    })


    pw.close()
  }

  def convertPOS(posFilename: String, name: String): Unit = {
    val tokenizer = new Tokenizer()

    val filename = "resources/evaluation/pos/" + name + ".txt"
    val pw = new PrintWriter(filename)
    var previous = true
    var sentence = ""
    Source.fromFile(posFilename).getLines()
      .foreach(line => {
        if (line.startsWith("#") && !previous) {
          pw.println(sentence)
          previous = true
          sentence = ""
        }
        else if (!line.startsWith("#") && line.nonEmpty) {
          previous = false
          val split = line.split("[\t\\s]+")
          val id = split.head
          if (!id.contains("_")) {
            sentence += " " + split(1) + "/" + split(3)
          }
        }
      })

    pw.close()
  }

  def convertSentiment(sentimentFilename: String, name: String): Unit = {


    val filename = "resources/evaluation/sentiment/" + name + ".txt"
    val pw = new PrintWriter(filename)
    val lines = Source.fromFile(sentimentFilename).getLines()

    lines.next()
    lines.map(line => {
        val split1 = line.split("\"\\,")
        val split = if (split1.length == 2) {
          val split11 = split1(1).split("\\,")
          Array(split1(0), split11(0), split11(1))
        }
        else {
          line.split("\\,")
        }

        val item = split.head
        val input = (if (item.startsWith("\"")) item.substring(1)
        else item)
        (input, split)
      }).filter(pair => {
        pair._2.length == 3
      }).map(pair => (pair._1, pair._2(1)))
      .foreach(pair => {
        val line = pair._1.replaceAll("\t", " ") + "\t" + pair._2
        pw.println(line)
      })

    pw.close()
  }

}

object DatasetConversion extends DatasetConversion() {
  def main(args: Array[String]): Unit = {
    /*convertNER("resources/evaluation/ner/train.json","train")
    convertNER("resources/evaluation/ner/test.json","test")
    convertPOS("resources/evaluation/pos/boun-train.conllu","train")
    convertPOS("resources/evaluation/pos/boun-test.conllu","test")
    convertSentiment("resources/evaluation/sentiment/train.csv", "train")
    convertSentiment("resources/evaluation/sentiment/test.csv", "test")
*/
    //createNERVocabulary()
    createSentimentVocabulary()
    //createPOSVocabulary()
  }
}
