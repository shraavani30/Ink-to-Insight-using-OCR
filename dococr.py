from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import matplotlib.pyplot as plt

# Load the OCR model
model = ocr_predictor(pretrained=True)

# Load the PDF document
doc = DocumentFile.from_pdf("png2pdf.pdf")

# Analyze the document to detect and predict words
result = model(doc)

# Extract words into a string with spaces
all_words = []
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                all_words.append(word.value)  # Append each detected word to the list

# Create the formatted output where every 7 words are on a new line
formatted_output = ''
for i in range(0, len(all_words), 7):
    formatted_output += ' '.join(all_words[i:i+7]) + '\n'  # Add newline after every 7 words

# Write the result into a text file
with open("original.txt", "w") as file:
    file.write(formatted_output)

# Optionally, you can still display the result with bounding boxes
result.show()
