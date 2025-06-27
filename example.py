import requests

# Define the API endpoint and your API key
api_key = 'K85357866588957'
url_api = 'https://api.ocr.space/parse/image'

# Path to your document (PDF or image)
file_path = 'page_1.pdf'  # Replace with the actual file path

# Prepare the payload and headers
with open(file_path, 'rb') as f:
    payload = {
        'apikey': api_key,
        'language': 'eng',  # You can change this to the required language, like 'hin' for Hindi, 'guj' for Gujarati
        'ocrengine': 2,  # Try changing this to 1 or 2
        'isOverlayRequired': True # Set to True if you want the OCR overlay on the image
    }

    # Upload the file to the API
    response = requests.post(url_api, files={'file': f}, data=payload)

# Check the response and parse the JSON result
if response.status_code == 200:
    result = response.json()
    # Extracting the parsed text from the JSON result
    parsed_text = result.get('ParsedResults', [])[0].get('ParsedText', '')
    print("Parsed Text:", parsed_text)
else:
    print(f"Error: {response.status_code}")
output_file_path = 'original.txt'

# Write the parsed text to the output file
with open(output_file_path, 'w') as file:
    file.write(parsed_text)

print(f"Text written to {output_file_path} successfully.")
