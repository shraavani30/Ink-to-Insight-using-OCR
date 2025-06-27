# PyTorch library is used for building and training neural networks, and it's particularly useful for natural language processing tasks.
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a smaller pre-trained T5-small model and tokenizer to improve performance
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Using smaller model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU

# Move the model to the device (GPU or CPU)
model.to(device)

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to generate a summary
def generate_summary(text):
    # Preprocess the input text
    preprocessed_text = text.strip().replace('\n', '')
    t5_input_text = 'summarize: ' + preprocessed_text[:512]  # Limiting input to 1024 characters

    # Tokenize the input text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt').to(device)

    # Generate summary with reduced beam size and adjusted length
    summary_ids = model.generate(tokenized_text,
                                 num_beams=2,  # Reduce beams for faster performance
                                 no_repeat_ngram_size=2,
                                 min_length=100,  # Adjust minimum length
                                 max_length=200,  # Adjust maximum length
                                 early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Split the summary into words and add a new line after every 7 words
    words = summary.split()
    formatted_summary = '\n'.join([' '.join(words[i:i + 7]) for i in range(0, len(words), 7)])

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

# Save the summary to a text file
output_file_path = 'abstractivesummary.txt'
with open(output_file_path, 'w') as file:
    file.write(summary)

print("\nSummary saved to", output_file_path)
