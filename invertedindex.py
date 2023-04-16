import os
import math
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import PyPDF2
nltk.download('wordnet')

def get_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        # create a pdf reader object
        pdf_reader = PyPDF2.PdfReader(f)

        # get the total number of pages
        num_pages = len(pdf_reader.pages)

        # create an empty string to store the text
        text = ''

        # iterate through all the pages and extract the text
        for page_num in range(num_pages):
            # get the page object
            page = pdf_reader.pages[page_num]

            # extract the text from the page
            page_text = page.extract_text()

            # append the page text to the overall text
            text += page_text

    return text

# Read the vocabulary file
vocab = {}
with open('vocabulary.txt', 'r') as f:
    for line in f:
        index, term = line.strip().split()
        vocab[term] = int(index)

# Set up the preprocessing steps
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Read the corrupt files list
with open('corruptfiles.txt', 'r') as f:
    corrupt_files = set(line.strip() for line in f)

rawtermfreq = {}

# Iterate through each folder and its pdf files using a loop
for folder in ['administrative', 'civil', 'commercial', 'constitutional', 'criminal', 'environmental', 'family', 'tax']:
    for filename in os.listdir(folder):
        if filename.endswith('.pdf') and filename not in corrupt_files:
            # Use a PDF parser library like PyPDF2 or pdfminer to extract the text from the pdf files
            text = get_text_from_pdf(os.path.join(folder, filename))
            
            # Perform preprocessing steps like tokenization, removing punctuation marks and less than three-character words,
            # normalization, stemming, and lemmatization on the extracted text
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokens = [t for t in tokens if t not in string.punctuation and t.isalpha() and len(t) > 3 and t not in stop_words]
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            tokens = [stemmer.stem(t) for t in tokens]
            
            initial = ''
            if folder == 'administrative':
                initial = 'A'
            elif folder == 'civil':
                initial = 'B'
            elif folder == 'commercial':
                initial = 'C'
            elif folder == 'constitutional':
                initial = 'D'
            elif folder == 'criminal':
                initial = 'E'
            elif folder == 'environmental':
                initial = 'F'
            elif folder == 'family':
                initial = 'G'
            elif folder == 'tax':
                initial = 'H'

            # get document identification number
            doc_id = initial + filename.split("_")[0]

            # calculate term frequency for each term in the document
            term_freq = {}
            for token in tokens:
                if token in vocab:
                    term_index = vocab[token]
                    if term_index in term_freq:
                        term_freq[term_index] += 1
                    else:
                        term_freq[term_index] = 1
            
            rawtermfreq[doc_id] = term_freq

# Create raw terms frequency dictionary
rawfreq = {}
for term, index in sorted(vocab.items(), key=lambda x: x[1]):
    term_dict = {}
    for doc_id, freq_dict in sorted(rawtermfreq.items()):
        if index in freq_dict:
            term_freq = freq_dict[index]
            term_dict[doc_id] = term_freq
    if term_dict:
        rawfreq[index] = term_dict
    else:
        print(f"Warning: no term frequency data found for term '{term}'")

    # Sort document ids in term_dict
    term_dict = dict(sorted(term_dict.items()))

# Sort index numbers in rawfreq dictionary
rawfreq = dict(sorted(rawfreq.items()))

# Write rawfreq dictionary to file
with open('rawtermfreq.txt', 'w') as f:
    for index, term_dict in rawfreq.items():
        line = f"{index}"
        for doc_id, freq in sorted(term_dict.items()):
            line += f" {doc_id}:{freq}"
        f.write(line + "\n")

# Create logtermfreq dictionary
logtermfreq = {}
for term, index in sorted(vocab.items(), key=lambda x: x[1]):
    term_dict = {}
    for doc_id, freq_dict in sorted(rawtermfreq.items()):
        if index in freq_dict:
            term_freq = freq_dict[index]
            term_dict[doc_id] = 1 + math.log10(term_freq)
    if term_dict:
        logtermfreq[index] = term_dict
    else:
        print(f"Warning: no term frequency data found for term '{term}'")

    # Sort document ids in term_dict
    term_dict = dict(sorted(term_dict.items()))

# Sort index numbers in logtermfreq dictionary
logtermfreq = dict(sorted(logtermfreq.items()))

# Write logtermfreq dictionary to file
with open('logtermfreq.txt', 'w') as f:
    for index, term_dict in logtermfreq.items():
        line = f"{index}"
        for doc_id, freq in sorted(term_dict.items()):
            line += f" {doc_id}:{freq}"
        f.write(line + "\n")

# # Read the raw frequency file
# rawfreq_file = "rawtermfreq.txt"
# rawfreq = {}
# with open(rawfreq_file, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         term_id = int(parts[0])
#         doc_freqs = {}
#         for df_part in parts[1:]:
#             doc_id, freq = df_part.split(":")
#             doc_freqs[doc_id] = int(freq)
#         rawfreq[term_id] = doc_freqs

# # Read the log term frequency file
# logtermfreq_file = "logtermfreq.txt"
# logtermfreq = {}
# with open(logtermfreq_file, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         term_id = int(parts[0])
#         doc_freqs = {}
#         for df_part in parts[1:]:
#             doc_id, freq = df_part.split(":")
#             doc_freqs[doc_id] = float(freq)
#         logtermfreq[term_id] = doc_freqs

# total number of documents in the corpus
num_docs = 8803

# initialize a dictionary to store IDF values for each term
idf = {}

for term_id, doc_freqs in rawfreq.items():
    # doc_freqs is a dictionary with keys as document ids and values as raw frequency of the term in the document
    df = len(doc_freqs) # number of documents containing the term
    idf[term_id] = math.log10(num_docs/df)


# initialize a dictionary to store TF-IDF values for each term in each document
tfidf = {}

# Calculate TF-IDF values for each term in each document
for term_id, doc_freqs in rawfreq.items():
    for doc_id, raw_freq in doc_freqs.items():
        tf = logtermfreq[term_id][doc_id] # log term frequency of the term in the document
        tfidf_val = tf * idf[term_id] # TF-IDF value of the term in the document
        # Store the TF-IDF value in the tfidf dictionary
        if term_id not in tfidf:
            tfidf[term_id] = {}
        tfidf[term_id][doc_id] = tfidf_val

# Write idf dictionary to file
with open('idf.txt', 'w') as f:
    for index, idfvalue in idf.items():
        f.write('{} {}\n'.format(index, idfvalue))

# Write tfidfweighting dictionary to file
with open('tfidf.txt', 'w') as f:
    for index, term_dict in tfidf.items():
        line = f"{index}"
        for doc_id, freq in sorted(term_dict.items()):
            line += f" {doc_id}:{freq}"
        f.write(line + "\n")

bm25weight = {}
k = 0.5

for termindex, docfreq in rawfreq.items():
        for docid, freqofterm in docfreq.items():
            bm25 = ((k+1) * int(freqofterm)) / (int(freqofterm) + k)
            bm25 = bm25 * idf[termindex]
            if termindex not in bm25weight:
                bm25weight[termindex] = {}
            bm25weight[termindex][docid] = bm25

# Write bm25weighting dictionary to file
with open('bm25.txt', 'w') as f:
    for index, term_dict in bm25weight.items():
        line = f"{index}"
        for doc_id, freq in sorted(term_dict.items()):
            line += f" {doc_id}:{freq}"
        f.write(line + "\n")