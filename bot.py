from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import csv
import pickle
import sys


class Bot:
   def __init__(self, filename):
      self.knowledge = filename 
      self.label = []
      self.feature = []
      self.model_name = "model.iro"
         
      reader = csv.reader(open(self.knowledge, 'r'))
      for data in reader:
         try:
            self.label.append(data[1]) # feature on 1st index
            self.feature.append(data[0]) #label on 2nd index
         except:
            continue
      
      # run feature extraction using CountVectorizer
   
      self.tf = TfidfVectorizer()

   def train(self):
      print('training bot with %s'%self.knowledge)
         
      # run feature extraction using CountVectorizer
      #cv = CountVectorizer()
      tr = self.tf.fit_transform(self.feature)
      clf  = LinearSVC()
      clf.fit(tr, self.label)
      pickle.dump(clf, open(self.model_name, 'wb'))
      print("bot has been trained...!!!")

   def predict(self, cmd):
      clf = pickle.load(open(self.model_name, 'rb'))
      x = self.tf.transform(cmd.split())
      return clf.predict(x)[0]


if __name__ == "__main__":
   bot = Bot('data.csv')
   bot.train()
   
   while True:
      w = input(" cmd > ")
      print("input: %s"%w)
      print(bot.predict(w))

