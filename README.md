# Hygieia-Mini: Medical Diagnostics Web App

Hygieia-Mini is a Flask-based web application for medical diagnostics, featuring:
- Dermatology image analysis (skin lesion classification)
- Diabetes risk assessment (symptom-based)

## Features
- Upload skin images for dermatology predictions
- Fill out a form for diabetes risk assessment
- Results displayed in a user-friendly web interface

## Requirements
- Python 3.10 or 3.11 recommended
- pip (Python package manager)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Krishna-Vijay-G/Hygieia-mini.git
   cd Hygieia-Mini
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or place required model/data files:**
   - Place the required model files in the `models/` directory as described in the project structure.

5. **Run the app locally:**
   ```bash
   python app.py
   ```
   The app will be available at [http://localhost:8000](http://localhost:8000)

## Project Structure
```
app.py
model_bridge.py
derm_model.py
diab_model.py
requirements.txt
models/
  Dermatology_Model/
  Diabetes_Model/
static/
templates/
uploads/
```

## Usage
- Visit the home page for navigation.
- Use the Dermatology page to upload an image and get predictions.
- Use the Diabetes page to fill out the form and get a risk assessment.

## License
This project is for educational and research purposes only.
