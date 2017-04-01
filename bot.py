from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

#import teleport
import csv
import pickle


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
      self.cv = CountVectorizer()
      self.tf = TfidfVectorizer()

   def train(self):
      print('training bot with %s'%self.knowledge)
         
      # run feature extraction using CountVectorizer
      #cv = CountVectorizer()
      tr = self.tf.fit_transform(self.feature)
      svc = SVC()
      svc.fit(tr, self.label)
      pickle.dump(svc, open(self.model_name, 'wb'))
      print("bot has been trained...!!!")

   def predict(self, cmd):
      clf = pickle.load(open(self.model_name, 'rb'))
      x = self.tf.transform(cmd.split())
      print(clf.predict(x)[0])

         
bot = Bot('data.csv')
bot.train()
bot.predict('link web')

