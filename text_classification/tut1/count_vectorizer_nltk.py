from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "How are you?",
    "I am fine!",
    "Thank you for asking :)",
    "let's see if this works out",
    "YES!!!!!"
]

# initialize CountVectorizer
ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

# fit the vectorizer on the corpus
ctv.fit(corpus)

corpus_transformed = ctv.transform(corpus)
print(ctv.vocabulary_)
print(corpus_transformed)
