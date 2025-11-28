import os
import cv2
import base64
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load YOLO model for license plate detection
print("Loading YOLO model...")
yolo_model = YOLO('license-plate-finetune-v1x.pt')

# Load TrOCR model for text recognition
print("Loading TrOCR model...")
ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
print("Models loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_plate(image, bbox):
    """Extract text from a cropped license plate region"""
    try:
        # Crop the license plate region
        x1, y1, x2, y2 = map(int, bbox)
        plate_img = image[y1:y2, x1:x2]
        
        # Convert BGR to RGB
        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(plate_img_rgb)
        
        # Preprocess and recognize text
        pixel_values = ocr_processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Error reading text"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image
        image = cv2.imread(filepath)
        
        # Run YOLO detection
        results = yolo_model.predict(source=filepath, conf=0.25)
        
        # Extract detections and recognize text
        detections = []
        for idx, box in enumerate(results[0].boxes):
            bbox = box.xyxy[0].tolist()
            
            # Extract text from the license plate
            plate_text = extract_text_from_plate(image, bbox)
            
            detection = {
                'id': idx + 1,
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
                'bbox': bbox,
                'text': plate_text
            }
            detections.append(detection)
        
        # Save annotated result image
        result_filename = f'result_{filename}'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        annotated_img = results[0].plot()
        
        # Add recognized text to the image
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            # Draw text background
            text_size = cv2.getTextSize(det['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(annotated_img, (x1, y2 + 5), (x1 + text_size[0] + 10, y2 + text_size[1] + 15), (0, 255, 0), -1)
            # Draw text
            cv2.putText(annotated_img, det['text'], (x1 + 5, y2 + text_size[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.imwrite(result_path, annotated_img)
        
        # Convert result image to base64
        with open(result_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Also save individual plate crops
        plate_images = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            plate_crop = image[y1:y2, x1:x2]
            plate_filename = f'plate_{idx}_{filename}'
            plate_path = os.path.join(app.config['RESULT_FOLDER'], plate_filename)
            cv2.imwrite(plate_path, plate_crop)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', plate_crop)
            plate_base64 = base64.b64encode(buffer).decode('utf-8')
            plate_images.append(f'data:image/jpeg;base64,{plate_base64}')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'result_image': f'data:image/jpeg;base64,{img_base64}',
            'plate_images': plate_images
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/result/<filename>')
def get_result(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)