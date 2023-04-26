import random

# Read the tfidf file
tfidf_file = "tfidf.txt"
tfidf = {}
with open(tfidf_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        term_id = int(parts[0])
        doc_freqs = {}
        for df_part in parts[1:]:
            doc_id, freq = df_part.split(":")
            doc_freqs[doc_id] = float(freq)
        tfidf[term_id] = doc_freqs

# dictionary to store tfidf values for each vocab term
vocab_tfidf = {}
for index, term_dict in tfidf.items():
    vocab_tfidf[index] = sum(term_dict.values())

sorted_tfidf_keys = sorted(vocab_tfidf, key=vocab_tfidf.get, reverse=True)[:10]
print(sorted_tfidf_keys)

# Read the bm25 file
bm25_file = "bm25.txt"
bm25 = {}
with open(bm25_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        term_id = int(parts[0])
        doc_freqs = {}
        for df_part in parts[1:]:
            doc_id, freq = df_part.split(":")
            doc_freqs[doc_id] = float(freq)
        bm25[term_id] = doc_freqs

# dictionary to store bm25 values for each vocab term
vocab_bm25 = {}
for index, term_dict in bm25.items():
    vocab_bm25[index] = sum(term_dict.values())

sorted_bm25_keys = sorted(vocab_bm25, key=vocab_bm25.get, reverse=True)[:10]
print(sorted_bm25_keys)

vocab_file = "vocabulary.txt"
vocab = {}

with open(vocab_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        term_id = int(parts[0])
        term = parts[1]
        vocab[term_id] = term

tfidf_terms = [vocab[key] for key in sorted_tfidf_keys]
bm25_terms = [vocab[key] for key in sorted_bm25_keys]

print(tfidf_terms)
print(bm25_terms)

# Create two-word queries
two_word_queries = []
for i in range(5):
    query = random.sample(tfidf_terms, 2)
    two_word_queries.append(query[0] + " " + query[1])

# Create three-term queries
three_term_queries = []
for i in range(5):
    query = random.sample(bm25_terms, 3)
    three_term_queries.append(query[0] + " " + query[1] + " " + query[2])

# Combine two-word and three-term queries to create final benchmark
benchmark = two_word_queries + three_term_queries

print(benchmark)

# Write benchmark to file
with open("benchmark.txt", "w") as f:
    f.write("Query Number\tQuery\n")
    for i, query in enumerate(benchmark):
        f.write(f"{i+1}\t{query}\n")
