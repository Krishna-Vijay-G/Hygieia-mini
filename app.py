from flask import Flask, Blueprint, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import sys
import uuid

from model_bridge import predict_dermatology, predict_diabetes

# Create Blueprint for Hygieia Models
hygieia_models = Blueprint('hygieia_models', __name__, 
                   template_folder='templates',
                   static_folder='static')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@hygieia_models.route('/')
def index():
    """Hygieia Models home page"""
    return render_template('index.html')

@hygieia_models.route('/dermatology', methods=['GET', 'POST'])
def dermatology():
    """Hygieia Models dermatology analysis"""
    if request.method == 'POST':
        print("DEBUG: POST request received")
        print("DEBUG: Files in request:", list(request.files.keys()))
        
        if 'image' not in request.files:
            print("DEBUG: No 'image' key in request.files")
            flash('No image file selected', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        print("DEBUG: File object:", file)
        print("DEBUG: Filename:", file.filename)
        
        if file.filename == '':
            print("DEBUG: Empty filename")
            flash('No image file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            print("DEBUG: File is valid")
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print("DEBUG: Saving to:", filepath)
            
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Save file
            file.save(filepath)
            print("DEBUG: File saved successfully")
            
            try:
                # Get prediction
                print("DEBUG: Starting prediction")
                prediction = predict_dermatology(filepath)
                print("DEBUG: Prediction successful:", prediction)
                
                # Render results
                return render_template('results.html', 
                                     result_type='dermatology',
                                     prediction=prediction,
                                     image_filename=filename)
                                     
            except Exception as e:
                print("DEBUG: Prediction failed:", str(e))
                flash(f'Analysis failed: {str(e)}', 'error')
                return redirect(request.url)
        else:
            print("DEBUG: File not allowed or invalid")
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(request.url)
    
    return render_template('dermatology.html')

@hygieia_models.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    """Hygieia Models diabetes risk assessment using UCI symptom-based model"""
    if request.method == 'POST':
        try:
            # Get form data for UCI symptom-based model
            # Build input dict with UCI symptom features
            data = {
                'age': request.form.get('age', ''),
                'gender': request.form.get('gender', ''),
                'polyuria': request.form.get('polyuria', ''),
                'polydipsia': request.form.get('polydipsia', ''),
                'sudden_weight_loss': request.form.get('sudden_weight_loss', ''),
                'weakness': request.form.get('weakness', ''),
                'polyphagia': request.form.get('polyphagia', ''),
                'genital_thrush': request.form.get('genital_thrush', ''),
                'visual_blurring': request.form.get('visual_blurring', ''),
                'itching': request.form.get('itching', ''),
                'irritability': request.form.get('irritability', ''),
                'delayed_healing': request.form.get('delayed_healing', ''),
                'partial_paresis': request.form.get('partial_paresis', ''),
                'muscle_stiffness': request.form.get('muscle_stiffness', ''),
                'alopecia': request.form.get('alopecia', ''),
                'obesity': request.form.get('obesity', '')
            }

            # Convert age to integer
            try:
                data['age'] = int(data['age'])
            except ValueError:
                flash('Invalid age value. Please enter a valid number.', 'error')
                return redirect(request.url)
            
            # Validate age range
            if not (10 <= data['age'] <= 120):
                flash('Age must be between 10-120 years', 'error')
                return redirect(request.url)
            
            # Validate gender
            if data['gender'] not in ['Male', 'Female']:
                flash('Please select a valid gender', 'error')
                return redirect(request.url)
            
            # Validate all symptom fields are Yes or No
            symptom_fields = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
                            'polyphagia', 'genital_thrush', 'visual_blurring', 'itching',
                            'irritability', 'delayed_healing', 'partial_paresis', 
                            'muscle_stiffness', 'alopecia', 'obesity']
            
            for field in symptom_fields:
                if data[field] not in ['Yes', 'No']:
                    flash(f'Please select Yes or No for {field.replace("_", " ").title()}', 'error')
                    return redirect(request.url)
            
            # Get prediction from UCI symptom-based model (model_bridge.predict_diabetes)
            # This function expects UCI symptom features
            prediction = predict_diabetes(data)
            
            # Render results
            return render_template('results.html', 
                                 result_type='diabetes',
                                 prediction=prediction,
                                 input_data=data)
                                 
        except ValueError as e:
            flash('Invalid input values. Please check your entries.', 'error')
            return redirect(request.url)
        except Exception as e:
            flash(f'Assessment failed: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('diabetes.html')

@hygieia_models.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

def create_hygieia_models_app():
    """Create and configure the Hygieia Models Flask app"""
    app = Flask(__name__)
    app.secret_key = 'hygieia-models-key-' + str(uuid.uuid4())
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    # Register the blueprint
    app.register_blueprint(hygieia_models)
    
    return app

app = create_hygieia_models_app() 

if __name__ == '__main__':
    app = create_hygieia_models_app()
    print("Starting Hygieia Models Medical Diagnostics Site...")
    print("Visit: http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')
