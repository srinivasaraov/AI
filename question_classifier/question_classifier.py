import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
#from sonetel.ai.classify.question_classifier2 import training_data

class QuestionClassifier():
    
    def __init__(self, model_file):
        self.model = model_file
    
    def load_data(self, filename):
        res = []
        with open(filename, 'r') as f:
            for line in f:
                question, label = line.split(",,,", 1)
                res.append((question.strip(), label.strip()))
        return res
    
    def train(self, data):
        labels = [line[1] for line in data]
        #OPTION1: Use the whole question for training
        #training_data = [line[0] for line in data]
        
        #OPTION2: Use the first 3 words of the question
        training_data = []
        for line in data:
            tokens = line[0].split()
            tokens_begn = ""
            for i in range(0, 3):
                tokens_begn = tokens_begn + tokens[i] + " "

            training_data.append(tokens_begn.strip())
            
        
        clf = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,2))),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, 
                                                   random_state=42,max_iter=5, tol=None))
            ])
        
        clf.fit(training_data, labels)
        with open(self.model, 'wb') as f:
            pickle.dump(clf, f)
            
        print (clf.score(training_data, labels))

        #Cross validation
        res = cross_validation.cross_val_score(clf, training_data, labels, cv=10)
        print ("cross-validation accuracy")
        print (res)
        print ("Average:")
        print (sum(res) / float(len(res)))
    
    def predict_one(self, test):
        with open(self.model, 'rb') as f:
            clf = pickle.load(f)
        a = clf.predict([test])[0]
        return a
    
    def predict_many(self, test):
        with open(self.model, 'rb') as f:
            clf = pickle.load(f)
        a = clf.predict(test)
        return a
    
if __name__ == '__main__':
    classifier = QuestionClassifier("question_classifier.pkl")
    training_file = "./data/LabelledData.txt"
    test_file = "./data/test_data.txt"
    
    train_data = classifier.load_data(training_file)
    classifier.train(train_data)
    
    test_ques = "what time does the match start?"
    test_ques = "can you dim the lights?"
    test_ques = "how can you dim the lights?"
    test_ques = "when does the match start?"
    test_ques = "how do you know about that?"
    test_ques = "would this pick up pine needles on a lawn ?"
    label = classifier.predict_one(test_ques)
    print(test_ques, " ==> ", label)
    
    test_corpus = classifier.load_data(test_file)
    test_labels = [line[1] for line in test_corpus]
    test_data = [line[0] for line in test_corpus]
     
    test_predicted = classifier.predict_many(test_data)
     
    for item in test_data:
        print(item, "==>", classifier.predict_one(item))
 
    accuracy = accuracy_score(test_labels, test_predicted)
    print("Accuracy on test data")
    print(accuracy)
#     
