import os
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

#array of corrupt files
corruptfiles = []

# Create an empty dictionary to store the vocabulary terms and their index numbers
vocab = {}

# Initialize the index number
index = 0

# Set up the preprocessing steps
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Iterate through each folder and its pdf files using a loop
for folder in ['administrative', 'civil', 'commercial', 'constitutional', 'criminal', 'environmental', 'family', 'tax']:
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            try:
                # Use a PDF parser library like PyPDF2 or pdfminer to extract the text from the pdf files
                text = get_text_from_pdf(os.path.join(folder, filename))
                
                # Perform preprocessing steps like tokenization, removing punctuation marks and less than three-character words,
                # normalization, stemming, and lemmatization on the extracted text
                tokens = re.findall(r'\b\w+\b', text.lower())
                tokens = [t for t in tokens if t not in string.punctuation and len(t) > 2 and t not in stop_words]
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
                tokens = [stemmer.stem(t) for t in tokens]
                
                # For each token, check if it already exists in the dictionary. If not, add it to the dictionary with a unique index number.
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = index
                        index += 1
            except:
                corruptfiles.append(filename)

# Once all the pdf files in all the folders have been processed, write the vocabulary terms and their index numbers to a plain text file
with open('vocabulary.txt', 'w') as f:
    for term, index in vocab.items():
        try:
            f.write('{} {}\n'.format(index, term))
        except:
            pass

# store corrupt files names
with open('corruptfiles.txt', 'w') as f:
    for fn in corruptfiles:
        f.write('{}\n'.format(fn))
