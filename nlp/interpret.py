from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import RegexpParser, pos_tag, ne_chunk

from prediction import Prediction

temporary = """“Fundamentals have gone out the window and irrational exuberance may accelerate bitcoin past the $50K level ahead of the second quarter schedule,” Jehan Chu, managing partner at Hong Kong-based crypto investment firm Kenetic Capital, told CoinDesk."""

with open("articles/samples") as f:
  content = f.read()

btc_itr = ["bitcoin", "bitcoins", "btc", "xbt"]
btc_itr += [i.capitalize() for i in btc_itr] + [i.upper() for i in btc_itr]

patterns = {
  "direct_prediction0": """Chunk: {<NN.?><MD><VB.?>*<.*>*<CD>}""", # Bitcoin could hit $15000
  # "direct_prediction1": """Chunk: {<MD><VB.?><NN.?>*<.*>*<CD>}""", # May accelerate bitcoin past the $50k level
  # "direct_prediction2": """Chunk: {<MD>+<VB.?>+<NN.?>*<.*>*<CD>*<JJ>+<NN>}"""
}

def grabChunks(content):
  # Parse the article into sentences
  # Check each sentence for bitcoin NNP's.
  # If the sentence has a bitcoin NNP then check if it's a predictor statement.
  # Check if the prediction is the authors or the author is quoting someone
  # Return the prediction object

  raw_chunks = []

  tokens = sent_tokenize(content)
  for i in tokens:
    token = word_tokenize(i)
    pos_tagged = pos_tag(token)
    for i in patterns:
      chunker = RegexpParser(patterns[i])

      output = chunker.parse(pos_tagged)
      for subtree in output.subtrees(filter=lambda t: t.label() == 'Chunk'): # Looping through every chunk found
        raw_chunks.append(subtree)
      
  return raw_chunks

def compilePredictions(raw_chunks):
  predictions = []
  for chunk in raw_chunks:
    predictions.append(Prediction(raw_chunk=chunk))

  return predictions

if __name__ == "__main__":
  raw_chunks = grabChunks(temporary)
  for i in raw_chunks:
    print(i)