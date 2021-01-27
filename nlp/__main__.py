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

API_URL = "http://localhost:8080/api/"
API_KEY = "zYmotoXZixZvPpOEfrJSoakrMNwVABZcSMdGRWdILrQoBodpbP" # Used so only permitted users can make callbacks

class NLP():
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

  def __init__(self):
    self.articlePool = []
    self.predictionPool = []

    self.running = True

    self.chunkerPatterns = { # Prediction Statements
      "direct_prediction0": """Chunk: {<NN.?><MD><VB.?>*<.*>*<CD>}""", # E.g. Bitcoin could hit $15000
      "direct_prediction1": """Chunk: {<MD><VB.?><NN.?>*<.*>*<CD>}""", # E.g. May accelerate bitcoin past the $50k level
      "direct_prediction2": """Chunk: {<MD>+<VB.?>+<NN.?>*<.*>*<CD>*<JJ>+<NN>}"""
    }

    self.logger = logging.getLogger("eventLogger")

    self.createDB()

  def start(self):
    # Initializing threads
    
    threads = [
      threading.Thread(target=self.getArticles, daemon=True),
      threading.Thread(target=self.compilePredictions),
      threading.Thread(target=self.callback, daemon=True)
    ]

    for thread in threads:
      thread.start()

  def createDB(self):
    # Creating database for storing article ID's so predictions aren't made for the same article over and over. Database used for persistance.

    conn = sqlite3.connect("nlp.db")
    cur = conn.cursor()
    if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ARTICLE_ID'") == 0:
      cur.execute("""CREATE TABLE ARTICLE_ID(
        id TEXT NOT NULL );
      """)
    # Add table to store prediction ID's.

    conn.close()

  def callback(self, predictions):
    # Callback to API/ web server with analysis of article. Sends json dictionary of prediction objects.
    prediction_callback = requests.post(API_URL, data=predictions)

  def writeIndex(self, id):
    # Write article ID's to the article_id table.

    conn = sqlite3.connect("nlp.db")
    conn.execute("INSERT INTO ARTICLE_ID (id) \
              VALUES (?)", (id,))
    conn.commit()
    conn.close()

  def readIndexes(self):
    # Read all article ID's.

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
    # Grabs all articles from the API and converts them to Article objects, then adds them to articlePool for analysis.

    while self.running:
      articles = requests.get(API_URL + "articles/").json()
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
    # Created prediction objects for articles.

    predictions = []
    chunks = self.grabChunks(article.content)
    for chunk in chunks:
      prediction = Prediction(chunk=chunk, article=article, chunkerUsed=chunks[chunk])
      predictions.append(prediction)

    self.predictionPool.extend(predictions)

  def grabChunks(self, content):
    # Parse the article into sentences and filter each sentence for predictor statement chunks.

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