# PlateSight – License Plate Recognition

## ✅ Overview
PlateSight is a Python web application that detects vehicle license plates in images and reads the plate text using a deep-learning detector and OCR model.  
The app provides a simple web interface so users can upload an image and see the detected plate and recognized text.

---

## 🚀 Features
- **Web UI with Flask** for uploading images and viewing predictions.  
- **License plate detection** using a YOLO-style object detection model.  
- **Text recognition (OCR)** on the detected plate region to extract the alphanumeric plate number.  
- **Easy to run locally** with a virtual environment and `requirements.txt`.

---

## 🗂️ Project Structure
- `app.py` – Main Flask application (routes, model loading, inference pipeline).  
- `templates/index.html` – Frontend template for upload form and results display.  
- `requirements.txt` – Python dependencies for the project.  
- `venv/` – Local virtual environment (ignored by Git; do **not** commit).

---

## 🔧 Installation & Setup

### 1. Clone the repository
> git clone https://github.com/tharz-06/platesigh.git
> 
> cd platesigh

### 2. Create and activate virtual environment (Windows)
> python -m venv venv
> 
> venv\Scripts\activate

### 3. Install dependencies
> pip install -r requirements.txt


If the model weights file (for example `license-plate-finetune-v1x.pt`) is not included in the repo, download or place it in the project folder as expected by `app.py`.

---

## ▶️ Usage

### Run the Flask app
> venv\Scripts\activate
> 
> python app.py

Then open your browser and go to:

http://127.0.0.1:5000/

From the web page:

1. Upload an image containing a vehicle.  
2. Submit the form.  
3. View the detected license plate and the recognized plate text.

---

## 📌 Notes
- The `venv/` directory is intentionally excluded from version control using `.gitignore`. Each user should create their own virtual environment locally.  
- For best results, use clear images where the license plate is visible and not heavily blurred or occluded.








