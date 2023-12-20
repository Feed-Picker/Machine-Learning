import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def calculate_word_frequency(sentences):
    word_freq = FreqDist()
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
    return word_freq

def calculate_sentence_scores(word_freq, sentences):
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        score = 0
        for word in words:
            if word in word_freq:
                score += word_freq[word]
        sentence_scores[i] = score
    return sentence_scores

def calculate_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    return similarity_matrix

def sentence_similarity(sent1, sent2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sent1, sent2])
    similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
    return similarity[0][0]

def apply_text_rank(similarity_matrix, num_iters=100, d=0.85):
    scores = np.ones(len(similarity_matrix)) / len(similarity_matrix)

    for _ in range(num_iters):
        scores = (1 - d) + d * np.dot(similarity_matrix.T, scores)

    return scores

def generate_summary(sentences, scores, N=3):
    top_sentences = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    summary = [sentences[i] for i in top_sentences]
    return ' '.join(summary)