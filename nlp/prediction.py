class Prediction():
  def __init__(self, raw_chunk, price=[], exp_date=[], author=[]):
    self.price = price
    self.exp_date = exp_date
    self.author = author
    self.raw_chunk = raw_chunk

  def __str__(self):
    print(f"Price: {self.price}")
    print(f"Expiration Date: {self.exp_date}")
    print(f"Author: {self.author}")
    print(f"Raw Chunk: {self.raw_chunk}")