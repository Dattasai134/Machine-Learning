#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os

# Specify the input and output folder paths
input_folder = 'D:\Text extraction\Invoices'
output_folder = 'D:\Text extraction\Binary_Invoices'  # Create this folder if it doesn't exist

# Iterate through each image in the input folder
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)

    # Load the image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Option 3: Manual thresholding (adjust value as needed)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

    # Save the binarized image
    binary_filename = os.path.splitext(filename)[0] + '_binary.png'  # Add "_binary" before extension
    binary_path = os.path.join(output_folder, binary_filename)
    cv2.imwrite(binary_path, thresh)


# In[6]:


pip install pytesseract Pillow


# In[5]:


import pytesseract
from PIL import Image
import difflib

# Path to your Tesseract executable (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example path

# Path to your image
image_path = 'D:/Text extraction/Binary_Invoices/Invoice5_binary.png'

# Load the image
img = Image.open(image_path)

# Extract text using Tesseract
processed_text = pytesseract.image_to_string(img)

# Define the initial text (replace with the actual initial text if available)
initial_text = "Your initial text goes here"

# Print the extracted text
print("Processed Text:")
print(processed_text)

# Compare text accuracy
accuracy_score = difflib.SequenceMatcher(None, initial_text, processed_text).ratio()
print("\nText accuracy:", accuracy_score)


# In[ ]:




