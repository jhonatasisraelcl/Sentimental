from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

positive_texts = [
  "nós amamos você",
  "meu pai é muito legal",
  "sua mãe é do bem",
  "eles nos amam",
  "isso é bom",
  "ele é legal",
  "bom mesmo é fazer o bem",
  "elas amam pizza",
  "eu amo ele",
  "eu adoro ela",
  "ela adora feijão",
  "o bem sempre vence",
  "eu adoro quem me adora"
  "o amor é lindo"
]

negative_texts =  [
  "nós odiamos vocês", 
  "elas nos odeia",
  "você é mau",
  "ele é muito mau",
  "maria se deu mal", 
  "eles odeiam bebidas",
  "joão detesta cigarros",
  "ela odeia o meu perfume",
  "eu fui muito mal na prova",
  "falsidade é coisa ruim",
  "o que mata é a falsidade"
]

test_texts = [
  "eu adoro pizza",                 #positiva
  "meu professor é bom",            #positiva
  "sei que você odeia ela",         #negativa
  "algumas vezes ele é legal",      #positiva
  "eu estou muito mal",             #negativa
  "eu amo o feijão daqui",          #positiva
  "meu deus quanta falsidade"       #negativa
]

training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

vectorizer = CountVectorizer()
vectorizer.fit(training_texts)
print(vectorizer.vocabulary_)

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)

print("Classificação Automática")
print(classifier.predict(testing_vectors))

tree.export_graphviz(
    classifier,
    out_file='tree.dot',
    feature_names=vectorizer.get_feature_names(),
) 

def manual_classify(text):
    if "odeia" in text:
      return "negative"
    if "detesta" in text:
      return "negative"
    if "mau" in text:
      return "negative"
    if "mal" in text:
      return "negative"
    return "positive"

print("Classificação Manual")
predictions = []
for text in test_texts:
    prediction = manual_classify(text)
    predictions.append(prediction)
print(predictions)

print()
print("Comparativo Classificação Manual x Automática")
for i in range(len(test_texts)):
  print("%s"%test_texts[i])
  print("Man%15s"%predictions[i])
  print("Auto%14s"%classifier.predict(testing_vectors)[i])
  print()