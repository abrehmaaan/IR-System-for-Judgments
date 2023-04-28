from flask import Flask, render_template, request
import os
import math
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import json

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/search')
def search():
    query_text = request.args.get('q')
    # Load the normal tf-idf weights from file
    normalized_tfidf_dict = {}
    with open(os.path.dirname(__file__) + "/../normaltfidf.txt") as f:
        for line in f:
            parts = line.strip().split()
            term_id = int(parts[0])
            normalized_tfidf_dict[term_id] = {}
            for pair in parts[1:]:
                doc_id, normalized_tfidf_dict_val = pair.split(":")
                normalized_tfidf_dict[term_id][doc_id] = float(normalized_tfidf_dict_val)
    # Read the vocabulary file
    vocab = {}
    with open(os.path.dirname(__file__) + '/../vocabulary.txt', 'r') as f:
        for line in f:
            index, term = line.strip().split()
            vocab[term] = int(index)

    # Set up the preprocessing steps
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Perform preprocessing steps like tokenization, removing punctuation marks and less than three-character words,
    # normalization, stemming, and lemmatization on the extracted text
    tokens = re.findall(r'\b\w+\b', query_text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t.isalpha() and len(t) > 3 and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]

    # Calculate term frequency (tf) for query
    tf_query = {}
    for term in tokens:
        term_index = vocab[term]
        if term_index in tf_query:
            tf_query[term_index] += 1
        else:
            tf_query[term_index] = 1

    tf_queries = {}
    tf_queries[1] = tf_query
    # Store terms frequencies in dictionary
    rawfreq = {}
    for term, index in sorted(vocab.items(), key=lambda x: x[1]):
        term_dict = {}
        for query_id, freq_dict in sorted(tf_queries.items()):
            if index in freq_dict:
                term_freq = freq_dict[index]
                term_dict[query_id] = term_freq
        if term_dict:
            rawfreq[index] = term_dict

    # Create logtermfreq dictionary
    logtermfreq = {}
    for term, index in sorted(vocab.items(), key=lambda x: x[1]):
        term_dict = {}
        for query_id, freq_dict in sorted(tf_queries.items()):
            if index in freq_dict:
                term_freq = freq_dict[index]
                term_dict[query_id] = 1 + math.log10(term_freq)
        if term_dict:
            logtermfreq[index] = term_dict


    # Read the idf from file
    idf = {}
    with open(os.path.dirname(__file__) + '/../idf.txt', 'r') as f:
        for line in f:
            index, value = line.strip().split()
            idf[int(index)] = float(value)

    # initialize a dictionary to store TF-IDF values for each term in each query
    tfidf_queries = {}

    # Calculate TF-IDF values for each term in each query
    for term_id, query_freq in rawfreq.items():
        for query_id, raw_freq in query_freq.items():
            tf = logtermfreq[term_id][query_id] # log term frequency of the term in the query
            tfidf_val = tf * idf[term_id] # TF-IDF value of the term in the query
            # Store the TF-IDF value in the tfidf dictionary
            if term_id not in tfidf_queries:
                tfidf_queries[term_id] = {}
            tfidf_queries[term_id][query_id] = tfidf_val

    query_norm_dict = {}
    for term_id, query_dict in tfidf_queries.items():
        for query_id, tfidf_queries_val in query_dict.items():
            if query_id not in query_norm_dict:
                query_norm_dict[query_id] = 0
            query_norm_dict[query_id] += tfidf_queries_val ** 2

    for query_id, norm_val in query_norm_dict.items():
        query_norm_dict[query_id] = math.sqrt(norm_val)

    normalized_tfidf_queries_dict = {}
    for term_id, query_dict in tfidf_queries.items():
        normalized_tfidf_queries_dict[term_id] = {}
        for query_id, tfidf_queries_val in query_dict.items():
            normalized_tfidf_queries_dict[term_id][query_id] = tfidf_queries_val / query_norm_dict[query_id]

    query_ids = [1]

    normalized_tfidf_queries = {}

    for term_id, query_dict in tfidf_queries.items():
        for query_id, tfidf_queries_val in query_dict.items():
            if query_id not in normalized_tfidf_queries:
                normalized_tfidf_queries[query_id] = {}
            normalized_tfidf_queries[query_id][term_id] = normalized_tfidf_queries_dict[term_id][query_id]

    doc_filenames = {}
    with open(os.path.dirname(__file__) + '/../newfileinfo.txt', 'r') as f:
        for line in f:
            ind, fn = line.strip().split('\t')
            doc_filenames[ind] = fn

    doc_abstracts = {}
    with open(os.path.dirname(__file__) + '/../newextracteddata.txt', 'r') as f:
        for line in f:
            ind, fn = line.strip().split('\t')
            doc_abstracts[ind] = fn

    similarities = {}

    for query_id in query_ids:
        similarities[query_id] = {}
        for doc_id in doc_filenames.keys():
            similarities[query_id][doc_id] = 0

    for query_id, querytfidf in normalized_tfidf_queries.items():
        for term_index, tfidf_val in querytfidf.items():
            for doc_id, tfidf_val_doc in normalized_tfidf_dict[term_index].items():
                if doc_id in doc_filenames.keys():
                    similarities[query_id][doc_id] += tfidf_val * tfidf_val_doc

    top_docs = dict(sorted({k: v for k, v in similarities[1].items() if v != 0}.items(), key=lambda x: -x[1]))

    doc_details = []
    for doc_id, sim_score in top_docs.items():
        doc_details.append({'id': doc_id, 'score': sim_score, 'name': doc_filenames[doc_id], 'abstract': doc_abstracts[doc_id]})
    
    # Pass data to template and render it
    return render_template('search.html', results=doc_details)


if __name__ == "__main__":
    app.run(debug=True)
