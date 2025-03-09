

from flask import Flask, request, jsonify
import cv2
import pytesseract
import re
import tempfile
import os
import numpy as np

app = Flask(__name__)


def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
    img = cv2.imread(image_path)

    img = cv2.resize(img, None, fx=1.2, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )

    return thresh

def parse_nic_data(text):
    """Extract NIC details: full name, NIC number, and date of birth from OCR text."""
    data = {}

    # Normalize and clean text
    text = text.replace("\n", " ").replace(":", " ").strip()

    # Extract NIC Number (Handles both old and new formats)
    nic_match = re.search(r'\b\d{9}[Vv]\b|\b\d{12}\b', text)
    if nic_match:
        data['national_id'] = nic_match.group()

    # Extract Full Name
    name_match = re.search(r'Name\s*([A-Za-z\s]+(?:\s+[A-Za-z\s]+)*)', text)

    if name_match:
        data['full_name'] = name_match.group(1).strip()

    # Extract Date of Birth (DD/MM/YYYY or YYYY/MM/DD)
    dob_match = re.search(r'\b(\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2})\b', text)
    if dob_match:
        data['date_of_birth'] = dob_match.group(1)

    return data


@app.route('/extract', methods=['POST'])
def extract():
    """Handle OCR extraction request."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        # Preprocess the image
        processed_img = preprocess_image(temp_path)
        cv2.imwrite(temp_path, processed_img)  # Save for OCR

        # Perform OCR
        text = pytesseract.image_to_string(temp_path)
        print("OCR Text:", text)  # Debug: Print OCR text

        # Parse data from OCR text
        result = parse_nic_data(text)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            os.remove(temp_path)  # Delete temp file
        except PermissionError:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)