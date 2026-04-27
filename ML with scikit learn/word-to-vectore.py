import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph = """I have three visions for India..."""  # truncated for brevity

# Text preprocessing
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)

# ✅ Vocabulary access in Gensim 4.x
words = model.wv.index_to_key   # list of all words in vocab
print("Vocabulary:", words[:20])  # print first 20 words

# Safe lookup function
def safe_vector_lookup(word):
    if word in model.wv.key_to_index:
        print(f"Vector for '{word}':", model.wv[word])
        print(f"Similar to '{word}':", model.wv.most_similar(word))
    else:
        print(f"'{word}' not in vocabulary")

# Try words
safe_vector_lookup('war')
safe_vector_lookup('freedom')
safe_vector_lookup('vikram')
safe_vector_lookup('son')
safe_vector_lookup('india')
