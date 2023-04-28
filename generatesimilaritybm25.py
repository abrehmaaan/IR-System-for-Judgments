import math
# Tokenize queries using NLTK tokenizer
from nltk.tokenize import word_tokenize

# Load the tf-idf weights from file
bm25 = {}
with open("bm25.txt") as f:
    for line in f:
        parts = line.strip().split()
        term_id = int(parts[0])
        bm25[term_id] = {}
        for pair in parts[1:]:
            doc_id, bm25_val = pair.split(":")
            bm25[term_id][doc_id] = float(bm25_val)

doc_norm_dict = {}
for term_id, doc_dict in bm25.items():
    for doc_id, bm25_val in doc_dict.items():
        if doc_id not in doc_norm_dict:
            doc_norm_dict[doc_id] = 0
        doc_norm_dict[doc_id] += bm25_val ** 2

for doc_id, norm_val in doc_norm_dict.items():
    doc_norm_dict[doc_id] = math.sqrt(norm_val)

normalized_bm25_dict = {}
for term_id, doc_dict in bm25.items():
    normalized_bm25_dict[term_id] = {}
    for doc_id, bm25_val in doc_dict.items():
        normalized_bm25_dict[term_id][doc_id] = bm25_val / doc_norm_dict[doc_id]

# Write normalized bm25weighting dictionary to file
with open('normalbm25.txt', 'w') as f:
    for index, term_dict in normalized_bm25_dict.items():
        line = f"{index}"
        for doc_id, freq in sorted(term_dict.items()):
            line += f" {doc_id}:{freq}"
        f.write(line + "\n")

# Read the vocabulary file
vocab = {}
with open('vocabulary.txt', 'r') as f:
    for line in f:
        index, term = line.strip().split()
        vocab[term] = int(index)

# Load the queries from file
queries = {}
with open("benchmark.txt") as f:
    next(f) # ignore first line
    for line in f:
        parts = line.strip().split("\t")
        query_id = int(parts[0])
        query_text = parts[1]
        queries[query_id] = query_text

tokenized_queries = {}
for query_id, query_text in queries.items():
    tokenized_queries[query_id] = word_tokenize(query_text)

# Calculate term frequency (tf) for each query
tf_queries = {}
for query_id, tokens in tokenized_queries.items():
    tf_query = {}
    for term in tokens:
        term_index = vocab[term]
        if term_index in tf_query:
            tf_query[term_index] += 1
        else:
            tf_query[term_index] = 1
    tf_queries[query_id] = tf_query

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

    # Sort query ids in term_dict
    term_dict = dict(sorted(term_dict.items()))

# Sort index numbers in rawfreq dictionary
rawfreq = dict(sorted(rawfreq.items()))

# # Create logtermfreq dictionary
# logtermfreq = {}
# for term, index in sorted(vocab.items(), key=lambda x: x[1]):
#     term_dict = {}
#     for query_id, freq_dict in sorted(tf_queries.items()):
#         if index in freq_dict:
#             term_freq = freq_dict[index]
#             term_dict[query_id] = 1 + math.log10(term_freq)
#     if term_dict:
#         logtermfreq[index] = term_dict

#     # Sort document ids in term_dict
#     term_dict = dict(sorted(term_dict.items()))

# # Sort index numbers in logtermfreq dictionary
# logtermfreq = dict(sorted(logtermfreq.items()))

# # Read the idf from file
# idf = {}
# with open('idf.txt', 'r') as f:
#     for line in f:
#         index, value = line.strip().split()
#         idf[int(index)] = float(value)

# # initialize a dictionary to store TF-IDF values for each term in each query
# tfidf_queries = {}

# # Calculate TF-IDF values for each term in each query
# for term_id, query_freq in rawfreq.items():
#     for query_id, raw_freq in query_freq.items():
#         tf = logtermfreq[term_id][query_id] # log term frequency of the term in the query
#         tfidf_val = tf * idf[term_id] # TF-IDF value of the term in the query
#         # Store the TF-IDF value in the tfidf dictionary
#         if term_id not in tfidf_queries:
#             tfidf_queries[term_id] = {}
#         tfidf_queries[term_id][query_id] = tfidf_val

query_norm_dict = {}
for term_id, query_dict in rawfreq.items():
    for query_id, tf_queries_val in query_dict.items():
        if query_id not in query_norm_dict:
            query_norm_dict[query_id] = 0
        query_norm_dict[query_id] += tf_queries_val ** 2

for query_id, norm_val in query_norm_dict.items():
    query_norm_dict[query_id] = math.sqrt(norm_val)

normalized_queries_dict = {}
for term_id, query_dict in rawfreq.items():
    normalized_queries_dict[term_id] = {}
    for query_id, tf_queries_val in query_dict.items():
        normalized_queries_dict[term_id][query_id] = tf_queries_val / query_norm_dict[query_id]

query_ids = list(set([query_id for term_index in normalized_queries_dict for query_id in normalized_queries_dict[term_index]]))

normalized_queries = {}

for term_id, query_dict in rawfreq.items():
    for query_id, tfidf_queries_val in query_dict.items():
        if query_id not in normalized_queries:
            normalized_queries[query_id] = {}
        normalized_queries[query_id][term_id] = normalized_queries_dict[term_id][query_id]

doc_filenames = {}
with open('newfileinfo.txt', 'r') as f:
    for line in f:
        ind, fn = line.strip().split('\t')
        doc_filenames[ind] = fn

similarities = {}

for query_id in query_ids:
    similarities[query_id] = {}
    for doc_id in doc_filenames.keys():
        similarities[query_id][doc_id] = 0

for query_id, querytf in normalized_queries.items():
    for term_index, tf_val in querytf.items():
        for doc_id, bm25_val_doc in normalized_bm25_dict[term_index].items():
            if doc_id in doc_filenames.keys():
                similarities[query_id][doc_id] += tf_val * bm25_val_doc

with open("similaritiesbm25.txt", "w") as f:
    for query_id, doc_sim in similarities.items():
        f.write("Query Term {}\n".format(query_id))
        f.write("Weighting Scheme: BM25\n")
        for doc_id, similarity in doc_sim.items():
            f.write("{}\t{}\t{}\n".format(doc_id, doc_filenames[doc_id], similarity))
        f.write("\n")

top_similarities = {}

for query_id in similarities:
    doc_similarities = similarities[query_id]
    sorted_docs = sorted(doc_similarities.items(), key=lambda x: x[1], reverse=True)[:10]
    top_docs = {doc_id: similarity for doc_id, similarity in sorted_docs}
    top_similarities[query_id] = top_docs

with open("top10bm25.txt", "w") as f:
    for query_id, doc_sim in top_similarities.items():
        f.write("Query Term {}\n".format(query_id))
        f.write("Weighting Scheme: BM25\n")
        for doc_id, similarity in doc_sim.items():
            f.write("{}\t{}\t{}\n".format(doc_id, doc_filenames[doc_id], similarity))
        f.write("\n")