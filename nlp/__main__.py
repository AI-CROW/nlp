import threading
import logging
import time
import requests
import os
import sqlite3

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import RegexpParser, pos_tag, ne_chunk

from nlp.prediction import Prediction
from nlp.article import Article

class NLP():
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
  
  def __init__(self):
    self.articlePool = []
    self.predictionPool = []

    self.running = True

    self.chunkerPatterns = {
      "direct_prediction0": """Chunk: {<NN.?><MD><VB.?>*<.*>*<CD>}""", # Bitcoin could hit $15000
      "direct_prediction1": """Chunk: {<MD><VB.?><NN.?>*<.*>*<CD>}""", # May accelerate bitcoin past the $50k level
      "direct_prediction2": """Chunk: {<MD>+<VB.?>+<NN.?>*<.*>*<CD>*<JJ>+<NN>}"""
    }

    self.logger = logging.getLogger("eventLogger")

    self.createDB()

  def start(self):
    threads = [
      threading.Thread(target=self.getArticles, daemon=True),
      threading.Thread(target=self.compilePredictions),
      threading.Thread(target=self.callback, daemon=True)
    ]

    for thread in threads:
      thread.start()

  def createDB(self):
    conn = sqlite3.connect("nlp.db")
    cur = conn.cursor()
    if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ARTICLE_ID'") == 0:
      cur.execute("""CREATE TABLE ARTICLE_ID(
        id TEXT NOT NULL );
      """)
    if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='PREDICTIONS'") == 0:
      cur.execute("""CREATE TABLE PREDICTIONS{
        id INT PRIMARY KEY NOT NULL,
        price REAL NOT NULL,
        expDate TEXT NOT NULL,
        article_id INT NOT NULL,
        chunk TEXT NOT NULL,
        chunkUsed TEXT NOT NULL""")

    conn.close()

  def callback(self):
    pass

  def writeIndex(self, id):
    conn = sqlite3.connect("nlp.db")
    conn.execute("INSERT INTO ARTICLE_ID (id) \
              VALUES (?)", (id,))
    conn.commit()
    conn.close()

  def writePredictions(self):
    for prediction in self.predictionPool:
      pass
    
  def readIndexes(self):
    indexed_ids = []

    conn = sqlite3.connect("nlp.db")
    try:
      cursor = conn.execute("SELECT id FROM ARTICLE_ID")
      for row in cursor:
        indexed_ids.append(row[0])
    except sqlite3.OperationalError:
      self.logger.info("SQLite3 operational error.")
      time.sleep(1)

    conn.close()

    return indexed_ids

  def getArticles(self):
    while self.running:
      articles = requests.get(f"http://localhost:8080/api/articles/").json()
      count = 0
      for article_ in articles:
        if not article_["id"] in self.readIndexes():
          article = Article(
            id = article_["id"],
            title = article_["title"],
            postDate = article_["postDate"],
            content = article_["content"],
            url = article_["url"],
            site_id = article_["site_id"],
            author_id = article_["author_id"]
          )
          self.articlePool.append(article)
          count += 1

      self.logger.info(f"{count} articles added to pool.")
      time.sleep(5)

  def compilePredictions(self):
    while self.running:
      for article in self.articlePool:
        self.compilePrediction(article)
        self.articlePool.remove(article)
        self.writeIndex(article.id)

      self.logger.info(f"{len(self.articlePool)} predictions added to pool.")
      time.sleep(5)

  def compilePrediction(self, article):
    predictions = []
    chunks = self.grabChunks(article.content)
    for chunk in chunks:
      prediction = Prediction(chunk=chunk, article=article, chunkerUsed=chunks[chunk])
      predictions.append(prediction)

    return predictions

  def grabChunks(self, content):
    # Parse the article into sentences
    # Check each sentence for bitcoin NNP's.
    # If the sentence has a bitcoin NNP then check if it's a predictor statement.
    # Check if the prediction is the authors or the author is quoting someone
    # Return the prediction object

    rawChunks = {}

    tokens = sent_tokenize(content)
    for i in tokens:
      token = word_tokenize(i)
      pos_tagged = pos_tag(token)
      for i in self.chunkerPatterns:
        chunker = RegexpParser(self.chunkerPatterns[i])

        output = chunker.parse(pos_tagged)
        for subtree in output.subtrees(filter=lambda t: t.label() == "Chunk"): # Looping through every chunk found
          rawChunks[str(subtree)] = i
        
    return rawChunks  

def entry_point():
  try:
    nlp = NLP()
    nlp.start()
  except Exception as e:
    print(e)

if __name__ == "__main__":
  entry_point()