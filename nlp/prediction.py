class Prediction():
  def __init__(self, chunk, article, chunkerUsed, price=[], expDate=[]):
    self.price = price
    self.expDate = expDate
    self.article = article
    self.chunk = chunk
    self.chunkerUsed = chunkerUsed

  def __str__(self):
    print(f"Price: {self.price}")
    print(f"Expiration Date: {self.expDate}")
    print(f"Article: {self.article}")
    print(f"Raw Chunk: {self.chunk}")
    print(f"Chunker Used: {self.chunkerUsed}")