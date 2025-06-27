import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation  # Imported punctuation from string module
import pandas as pd
from heapq import nlargest

# Load the pre-trained Spacy model
nlp = spacy.load('en_core_web_sm')

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to generate a summary
def generate_summary(text):
    docx = nlp(text)

    # Get the tokens
    mytokens = [token.text for token in docx]

    # Get the stopwords and punctuation
    stopwords = list(STOP_WORDS)
    punctuations = punctuation + '\n'  # Renamed local variable to 'punctuations'

    # Calculate the word frequency
    word_freq = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in punctuations:  # Use 'punctuations' here
                if word.text not in word_freq.keys():
                    word_freq[word.text] = 1
                else:
                    word_freq[word.text] += 1

    # Normalize the word frequency
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word] / max_freq)

    # Get the sentence tokens
    sentence_tokens = [sent for sent in docx.sents]

    # Calculate the sentence scores
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_freq[word.text.lower()]
                else:
                    sentence_scores[sent] += word_freq[word.text.lower()]

    # Get the top sentences
    select_length = int(len(sentence_tokens) * 0.4)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # Create the final summary
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    # Format the summary to add a new line after every 10 words
    words = summary.split()
    formatted_summary = '\n'.join([' '.join(words[i:i+10]) for i in range(0, len(words), 10)])

    return formatted_summary

# Path to the input text file
input_file_path = 'original.txt'

# Read the text from the file
text = read_text_file(input_file_path)

# Generate the summary
summary = generate_summary(text)

# Print the summary
print("Summary:\n")
print(summary)

# Print the original text and its length
print("\nOriginal Text:")
print(text)
print("\nLength of Original Text:", len(text))
print("Length of Summary:", len(summary))

# Save the summary to a text file
output_file_path = 'extractivesummary.txt'
with open(output_file_path, 'w') as file:
    file.write(summary)

print("\nSummary saved to", output_file_path)
