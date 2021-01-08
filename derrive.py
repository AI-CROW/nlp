from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import RegexpParser, pos_tag

temporary = """Bitcoin will hit $15000 by January."""

BTC_ITR = ["bitcoin", "bitcoins", "btc", "xbt"]
BTC_ITR += [i.capitalize() for i in BTC_ITR] + [i.upper() for i in BTC_ITR]

with open("articles/example1") as f:
  content = f.read()

def interpretArticlePrediction(content):
  # Parse the article into sentences
  # Check each sentence for bitcoin NNP's.
  # If the sentence has a bitcoin NNP then check if it's a predictor statement.
  # Check if the prediction is the authors or the author is quoting someone
  # Return the prediction along with the predictors name

  pattern = """Chunk: {<NN.?><MD><VB.?>*<.*>*<CD>}"""
  chunker = RegexpParser(pattern)

  tokens = sent_tokenize(content)
  for i in tokens:
    token = word_tokenize(i)
    if any(item in BTC_ITR for item in token): # Sentence has Bitcoin NNP
      pos_tagged = pos_tag(token)
      output = chunker.parse(pos_tagged)
      output.draw()
      
if __name__ == "__main__":
  interpretArticlePrediction(temporary)
