import nltk
import random
nltk.download('punkt')
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Arabic stopwords (you may need to customize this list)
stopwords = ["و", "في", "من", "هذا", "هذه", "على", "إلى", "أن", "التي"]

# Sample positive and negative Arabic texts (you can replace these with your own data)
positive_text = "أنا سعيد جدًا بالنجاح في هذا المشروع."
negative_text = "لم يكن هناك أي تقدم في هذا المشروع."

# Tokenize the text into words
def tokenize(text):
    words = nltk.word_tokenize(text)
    return [word for word in words if word not in stopwords]

# Create labeled data with positive and negative examples
documents = [(tokenize(positive_text), 'positive'), (tokenize(negative_text), 'negative')]

# Shuffle the documents
random.shuffle(documents)

# Define a function to extract features from text (word presence/absence)
def document_features(document):
    features = {}
    for word in document:
        features['contains({})'.format(word)] = True
    return features

# Extract features from the documents
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the data into a training set and a test set (80% training, 20% testing)
train_set, test_set = featuresets[:int(len(featuresets) * 0.8)], featuresets[int(len(featuresets) * 0.8):]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test the classifier accuracy on the test set
accuracy_score = accuracy(classifier, test_set)
print(f"Classifier Accuracy: {accuracy_score:.2%}")

# Test the classifier on custom text
custom_text = "هذا المشروع يسير بسرعة ويبدو رائعًا."
custom_tokens = tokenize(custom_text)
custom_features = document_features(custom_tokens)
custom_category = classifier.classify(custom_features)
print(f"Custom Text Category: {custom_category}")


