
# Hygieia AI Model Overview

**Date:** November 19, 2025

## 1. Introduction

Hygieia is an AI-powered health screening application that integrates two distinct machine learning models to provide early-stage risk assessment for dermatological conditions and diabetes. This document provides a comprehensive overview of the architecture, training methodologies, and performance of these models.

The application is designed to be a user-friendly tool for preliminary screening, not a diagnostic tool. All predictions should be confirmed by a qualified healthcare professional.

## 2. The Hygieia Workflow

The Hygieia application provides a seamless user experience for accessing the two predictive models:

1.  **Model Selection:** The user chooses between the "Dermatology" and "Diabetes" screening tools from the main interface.
2.  **Data Input:**
    *   **Dermatology:** The user uploads a dermatoscopic image of a skin lesion.
    *   **Diabetes:** The user fills out a questionnaire detailing 16 clinical symptoms.
3.  **Backend Processing:** The web application backend, powered by Flask (`app.py`), receives the user's input.
4.  **Model Bridging:** A `model_bridge.py` component dynamically loads the appropriate pre-trained model (`.joblib` for the diabetes model, TensorFlow SavedModel for the dermatology model).
5.  **Prediction:** The selected model processes the input data and generates a risk prediction.
6.  **Results Display:** The prediction, along with a confidence score, is presented to the user on a results page.

## 3. Dermatology Model: Skin Lesion Classification

### 3.1. Model Architecture and Algorithm

*   **Algorithm:** Convolutional Neural Network (CNN)
*   **Base Model:** **MobileNetV2**, a lightweight and efficient deep learning architecture designed for mobile and resource-constrained environments.
*   **Technique:** **Transfer Learning**. The model leverages weights pre-trained on the large-scale ImageNet dataset, which provides a strong foundation for general image feature extraction. This base is then fine-tuned on a specialized medical dataset.

### 3.2. Dataset and Features

*   **Dataset:** **HAM10000 (Human Against Machine with 10000 training images)**. This dataset contains 10,015 dermatoscopic images of seven common pigmented skin lesion categories.
*   **Features:** The model learns hierarchical features directly from the image pixels. These features range from simple edges and textures in early layers to complex patterns and structures relevant to classifying skin lesions in deeper layers.

### 3.3. Training Methodology

1.  **Data Preprocessing:** Images are resized and normalized to match the input requirements of MobileNetV2.
2.  **Data Augmentation:** To prevent overfitting and improve generalization, the training data is augmented in real-time with random transformations such as rotation, shifting, and flipping.
3.  **Class Imbalance Handling:** The dataset is imbalanced. This was addressed using class weights during training, giving more importance to under-represented classes.
4.  **Fine-Tuning:** The top layers of the pre-trained MobileNetV2 were unfrozen and retrained on the HAM10000 dataset. This adapts the general-purpose feature extractor to the specific task of skin lesion classification.
5.  **Optimization:** The model was trained using the Adam optimizer with a learning rate scheduler to reduce the learning rate as training progressed, helping the model to converge to a more optimal solution.

### 3.4. Performance

*   **Accuracy:** **95.3%** on a held-out test set.
*   **Prediction Output:** The model outputs a probability distribution over the seven skin lesion categories, indicating the likelihood of the input image belonging to each class.

## 4. Diabetes Model: Symptom-Based Risk Prediction

### 4.1. Model Architecture and Algorithm

*   **Algorithm:** **LightGBM (Light Gradient Boosting Machine)**. This is a high-performance, tree-based gradient boosting framework.
*   **Key Advantages:** LightGBM is known for its speed, efficiency, and ability to handle categorical features directly, making it an excellent choice for this clinical dataset.

### 4.2. Dataset and Features

*   **Dataset:** **UCI Early Stage Diabetes Risk Prediction Dataset**. This dataset contains 520 patient records.
*   **Features:** The model uses **16 symptom-based binary features**. This is a critical aspect of the model; it relies on clinical signs observable by the patient rather than on laboratory test results.
    *   **Key Features:** `Polyuria` (frequent urination), `Polydipsia` (excessive thirst), `sudden weight loss`, `weakness`, `visual blurring`, etc.
    *   **Data Type:** Categorical (Yes/No), which are label-encoded into binary (1/0) format.

### 4.3. Training Methodology

1.  **Data Preprocessing:** Categorical features (e.g., 'Yes'/'No', 'Male'/'Female') were converted into numerical format using label encoding.
2.  **Class Imbalance Handling:** The dataset has an unequal number of positive and negative cases. This was managed by setting the `scale_pos_weight` parameter in the LightGBM classifier, which adjusts the weight of the positive class.
3.  **Model Training:** A LightGBM classifier was trained on an 80% split of the dataset.
4.  **Hyperparameter Optimization:** The model's hyperparameters (e.g., `num_leaves`, `learning_rate`, `n_estimators`) were tuned to achieve optimal performance on the validation set.

### 4.4. Performance

*   **Accuracy:** **98.1%** on a 20% held-out test set.
*   **AUC-ROC:** **1.000**, indicating perfect discrimination between the positive and negative classes on the test set. This means the model is exceptionally confident in its predictions.
*   **Inference Speed:** Extremely fast, at approximately **0.06ms per prediction**.

## 5. How Data is Predicted

### Dermatology Model Prediction Process

1.  **Input:** An uploaded image of a skin lesion.
2.  **Preprocessing:** The image is resized to the model's expected input size (e.g., 224x224 pixels) and its pixel values are normalized.
3.  **Inference:** The preprocessed image tensor is fed into the trained TensorFlow model (`derm_model.joblib` which is a saved MobileNetV2 model).
4.  **Output:** The model returns an array of probabilities, with each element corresponding to the predicted likelihood of one of the seven skin lesion classes. The class with the highest probability is chosen as the final prediction.

### Diabetes Model Prediction Process

1.  **Input:** A set of 16 binary (Yes/No) answers from the user's symptom questionnaire.
2.  **Preprocessing:** The categorical inputs are converted to their binary (1/0) encoded representation, matching the format used during training.
3.  **Inference:** The encoded feature vector is passed to the loaded LightGBM model (`diab_model.joblib`).
4.  **Output:** The model outputs a binary prediction (0 for 'No Risk', 1 for 'At Risk') and a confidence score representing the probability of the prediction being correct.

## 6. Conclusion

The Hygieia application successfully integrates two high-performing, yet architecturally distinct, machine learning models to address different healthcare screening challenges.

*   The **Dermatology Model** showcases the power of **deep learning and transfer learning** for image-based medical analysis, achieving high accuracy in a complex classification task.
*   The **Diabetes Model** highlights the effectiveness of **traditional machine learning algorithms like LightGBM** when applied to high-quality, clinically relevant, structured data, resulting in near-perfect prediction accuracy.

Both models are deployed within a user-friendly web application, demonstrating a practical and effective implementation of AI for accessible health screening.
