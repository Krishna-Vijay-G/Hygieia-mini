// Simple Medical Diagnostic Site JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const label = this.parentNode.querySelector('.file-upload-label');
            const fileName = e.target.files[0]?.name || 'Choose file...';
            
            if (e.target.files[0]) {
                label.innerHTML = `
                    <div class="file-upload-icon">✓</div>
                    <div>
                        <strong>Selected:</strong><br>
                        ${fileName}
                    </div>
                `;
                label.style.borderColor = '#27ae60';
                label.style.background = 'rgba(39, 174, 96, 0.1)';
            }
        });
    });

    // Make file upload labels clickable
    const fileUploadLabels = document.querySelectorAll('.file-upload-label');
    fileUploadLabels.forEach(label => {
        label.addEventListener('click', function() {
            const input = this.parentNode.querySelector('input[type="file"]');
            if (input) {
                input.click();
            }
        });
    });

    // Drag and drop functionality
    const fileUploadAreas = document.querySelectorAll('.file-upload');
    fileUploadAreas.forEach(area => {
        const input = area.querySelector('input[type="file"]');
        const label = area.querySelector('.file-upload-label');

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, unhighlight, false);
        });

        // Handle dropped files
        area.addEventListener('drop', handleDrop, false);

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        function highlight(e) {
            area.classList.add('drag-over');
        }

        function unhighlight(e) {
            area.classList.remove('drag-over');
        }

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length > 0 && files[0].type.startsWith('image/')) {
                input.files = files;
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                input.dispatchEvent(event);
            } else {
                showError('Please drop a valid image file (JPG, PNG, etc.)');
            }
        }
    });

    // Form submission handling
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            const loading = form.querySelector('.loading');
            
            if (submitBtn && loading) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Analyzing...';
                loading.style.display = 'block';
            }
        });
    });

    // Animate confidence bars
    const confidenceBars = document.querySelectorAll('.confidence-fill');
    confidenceBars.forEach(bar => {
        const confidence = bar.getAttribute('data-confidence');
        if (confidence) {
            setTimeout(() => {
                bar.style.width = confidence + '%';
            }, 500);
        }
    });

    // Add fade-in animation to elements
    const elements = document.querySelectorAll('.service-card, .form-container, .result-container');
    elements.forEach((el, index) => {
        setTimeout(() => {
            el.classList.add('fade-in');
        }, index * 200);
    });

    // Smooth scrolling for navigation
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Scroll to top for "#" links
    const hashLinks = document.querySelectorAll('a[href="#"]');
    hashLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    });

    // Form validation
    function validateDiabetesForm() {
        const form = document.querySelector('#diabetes-form');
        if (!form) return true;

        const requiredFields = form.querySelectorAll('input[required]');
        let isValid = true;

        requiredFields.forEach(field => {
            const value = parseFloat(field.value);
            const min = parseFloat(field.getAttribute('min'));
            const max = parseFloat(field.getAttribute('max'));

            if (isNaN(value) || value < min || value > max) {
                field.style.borderColor = '#e74c3c';
                isValid = false;
            } else {
                field.style.borderColor = '#ddd';
            }
        });

        return isValid;
    }

    // Real-time form validation
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.getAttribute('min'));
            const max = parseFloat(this.getAttribute('max'));

            if (isNaN(value) || value < min || value > max) {
                this.style.borderColor = '#e74c3c';
            } else {
                this.style.borderColor = '#27ae60';
            }
        });
    });

    // Image preview for dermatology
    const imageInput = document.querySelector('input[name="image"]');
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    // Create preview if doesn't exist
                    let preview = document.querySelector('.image-preview');
                    if (!preview) {
                        preview = document.createElement('div');
                        preview.className = 'image-preview';
                        preview.style.marginTop = '1rem';
                        preview.style.textAlign = 'center';
                        imageInput.parentNode.appendChild(preview);
                    }
                    
                    preview.innerHTML = `
                        <img src="${e.target.result}" 
                             style="max-width: 200px; max-height: 200px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                        <p style="margin-top: 0.5rem; color: #666;">Image Preview</p>
                    `;
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Add hover effects
    const cards = document.querySelectorAll('.service-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // BMI Calculator functionality (Inline version)
    const bmiCalculatorBtn = document.getElementById('bmi-calculator-btn');
    const bmiCalculatorSection = document.getElementById('bmi-calculator-section');
    const calculateBmiBtn = document.getElementById('calculate-bmi');
    const useBmiBtn = document.getElementById('use-bmi');
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiResult = document.getElementById('bmi-result');
    const bmiCategory = document.getElementById('bmi-category');
    const bmiFormInput = document.getElementById('bmi');

    // Toggle BMI calculator dropdown
    if (bmiCalculatorBtn) {
        bmiCalculatorBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (bmiCalculatorSection.style.display === 'none' || bmiCalculatorSection.style.display === '') {
                // Show calculator dropdown
                bmiCalculatorSection.style.display = 'block';
                bmiCalculatorBtn.classList.add('active');
                
                // Focus on height input
                if (heightInput) {
                    setTimeout(() => heightInput.focus(), 100);
                }
            } else {
                // Hide calculator dropdown
                bmiCalculatorSection.style.display = 'none';
                bmiCalculatorBtn.classList.remove('active');
            }
        });
    }

    // Calculate BMI
    if (calculateBmiBtn) {
        calculateBmiBtn.addEventListener('click', function() {
            const height = parseFloat(heightInput.value);
            const weight = parseFloat(weightInput.value);
            
            if (!height || !weight || height <= 0 || weight <= 0) {
                showError('Please enter valid height and weight values');
                return;
            }
            
            if (height < 50 || height > 250) {
                showError('Height must be between 50-250 cm');
                return;
            }
            
            if (weight < 20 || weight > 300) {
                showError('Weight must be between 20-300 kg');
                return;
            }
            
            // BMI formula: weight (kg) / [height (m)]²
            const heightInMeters = height / 100;
            const bmi = (weight / (heightInMeters * heightInMeters)).toFixed(1);
            
            // Display result
            bmiResult.textContent = bmi;
            
            // Determine category
            let category = '';
            let categoryColor = '';
            
            if (bmi < 18.5) {
                category = 'Underweight';
                categoryColor = '#3498db';
            } else if (bmi < 25) {
                category = 'Normal weight';
                categoryColor = '#27ae60';
            } else if (bmi < 30) {
                category = 'Overweight';
                categoryColor = '#f39c12';
            } else {
                category = 'Obese';
                categoryColor = '#e74c3c';
            }
            
            bmiCategory.textContent = category;
            bmiCategory.style.color = categoryColor;
            
            // Enable use button
            useBmiBtn.disabled = false;
            useBmiBtn.style.opacity = '1';
            
            // Show success message
            showSuccess('BMI calculated successfully!');
        });
    }

    // Use calculated BMI
    if (useBmiBtn) {
        useBmiBtn.addEventListener('click', function() {
            const bmiValue = bmiResult.textContent;
            if (bmiValue !== '--' && bmiValue !== '') {
                bmiFormInput.value = bmiValue;
                
                // Hide calculator dropdown
                bmiCalculatorSection.style.display = 'none';
                bmiCalculatorBtn.classList.remove('active');
                
                showSuccess('BMI value applied to form');
            }
        });
    }

    // Auto-calculate BMI when both fields have values
    if (heightInput && weightInput) {
        [heightInput, weightInput].forEach(input => {
            input.addEventListener('input', function() {
                const height = parseFloat(heightInput.value);
                const weight = parseFloat(weightInput.value);
                
                // Auto-enable calculate button when both fields have valid values
                if (height > 0 && weight > 0 && height >= 50 && height <= 250 && weight >= 20 && weight <= 300) {
                    calculateBmiBtn.disabled = false;
                    calculateBmiBtn.style.opacity = '1';
                } else {
                    calculateBmiBtn.disabled = true;
                    calculateBmiBtn.style.opacity = '0.6';
                }
            });
        });
    }
});

// Utility functions
function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success';
    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #d4edda;
        color: #155724;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    alert.textContent = message;
    document.body.appendChild(alert);
    
    setTimeout(() => {
        alert.remove();
    }, 3000);
}

function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-error';
    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #f8d7da;
        color: #721c24;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    alert.textContent = message;
    document.body.appendChild(alert);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Add slide in animation for alerts
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);