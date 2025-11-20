---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
language:
  - en
tags:
  - medical
  - dermatology
  - digital-dermatology
  - medical-embeddings
  - image-classification
  - image-feature-extraction
extra_gated_heading: Access Derm Foundation on Hugging Face
extra_gated_prompt: >-
  To access Derm Foundation on Hugging Face, you're required to review and
  agree to [Health AI Developer Foundation's terms of use](https://developers.google.com/health-ai-developer-foundations/terms).
  To do this, please ensure you're logged in to Hugging Face and click below.
  Requests are processed immediately.
extra_gated_button_content: Acknowledge license
library_name: derm-foundation
---

# Derm Foundation model card

**Model documentation**:
[Derm Foundation](https://developers.google.com/health-ai-developer-foundations/derm-foundation)

**Resources**:

*   Model on Google Cloud Model Garden:
    [Derm Foundation](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/derm-foundation)
*   Model on Hugging Face:
    [google/derm-foundation](https://huggingface.co/google/derm-foundation)
*   GitHub repository (supporting code, Colab notebooks, discussions, and
    issues): [derm-foundation](https://github.com/google-health/derm-foundation)
*   Quick start notebook:
    [notebooks/quick_start](https://github.com/google-health/derm-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb)
*   Support: See
    [Contact](https://developers.google.com/health-ai-developer-foundations/derm-foundation/get-started.md#contact).

**Terms of use**:
[Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms)

**Author**: Google

## Model information

This section describes the Derm Foundation model and how to use it.

### Description

Derm Foundation is a machine learning model designed to accelerate AI
development for skin image analysis for dermatology applications. It is
pre-trained on large amounts of labeled skin images to produce 6144 dimensional
embeddings that capture dense features relevant for analyzing these images. As a
result, Derm Foundation’s embeddings enable the efficient training of AI models
with significantly less data and compute than traditional methods.

### How to use

Following are some example code snippets to help you quickly get started running
the model locally. If you want to use the model at scale, we recommend that you
create a production version using
[Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/derm-foundation).

```python
# Download test image.
from PIL import Image
from io import BytesIO
from IPython.display import Image as IPImage, display
from huggingface_hub import from_pretrained_keras
import tensorflow as tf

# Download sample image
!wget -nc -q https://storage.googleapis.com/dx-scin-public-data/dataset/images/3445096909671059178.png

# Load the image
img = Image.open("3445096909671059178.png")
buf = BytesIO()
img.convert('RGB').save(buf, 'PNG')
image_bytes = buf.getvalue()

# Format input
input_tensor= tf.train.Example(features=tf.train.Features(
        feature={'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))
        })).SerializeToString()

# Load the model directly from Hugging Face Hub
loaded_model = from_pretrained_keras("google/derm-foundation")

# Call inference
infer = loaded_model.signatures["serving_default"]
output = infer(inputs=tf.constant([input_tensor]))

# Extract the embedding vector
embedding_vector = output['embedding'].numpy().flatten()
```

### Examples

See the following Colab notebooks for examples of how to use Derm Foundation:

*   To give the model a quick try, running it locally with weights from Hugging
    Face, see
    [Quick start notebook in Colab](https://colab.research.google.com/github/google-health/derm-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb).

*   For an example of how to use the model to train a linear classifier see
    [Linear classifier notebook in Colab](https://colab.research.google.com/github/google-health/derm-foundation/blob/master/notebooks/train_data_efficient_classifier.ipynb)

*   [DERM12345 Embeddings and Demo](https://github.com/abdurrahimyilmaz/derm12345_google-derm-foundation/tree/main) includes a demo using Derm Foundation precomputed embeddings for
    [DERM12345](https://www.nature.com/articles/s41597-024-04104-3). Special thanks to Abdurrahim Yilmaz for providing this.
  
### Model architecture overview

*   The model is a [BiT-M ResNet101x3](https://arxiv.org/abs/1912.11370).

Derm Foundation was trained in two stages. The first pre-training stage used
contrastive learning to train on a large number of public image-text pairs from
the internet. The image component of this pre-trained model was then fine-tuned
for condition classification and a couple other downstream tasks using a number
of clinical datasets (see below).

### Technical specifications

*   Model type: BiT-101x3 CNN (Convolutional Neural Network)
*   Key publications:
    *   BiT:
        [https://arxiv.org/abs/1912.11370](https://arxiv.org/abs/1912.11370)
    *   ConVIRT:
        [https://arxiv.org/abs/2010.00747](https://arxiv.org/abs/2010.00747)
*   Model created: 2023-12-19
*   Model version: Version: 1.0.0

### Performance and validation

Derm Foundation was evaluated for data-efficient accuracy across a range of
skin-related classifications tasks. Training a linear classifier on
Derm-Foundations embeddings were substantially more performant (10-15% increase
in accuracy) than doing the same for a standard BiT-M model across different
proportions of training data. See this
[Health-specific embedding tools for dermatology and pathology](https://research.google/blog/health-specific-embedding-tools-for-dermatology-and-pathology/)
for more details.

### Inputs and outputs

*   **Input**: `PNG` image file 448 x 448 pixels

*   **Output**: Embedding vector of floating point values (Dimensions: 6144)

## Dataset details

### Training dataset

Derm Foundation was trained in two stages. The first pre-training stage used
contrastive learning to train on a large number of public image-text pairs from
the internet. The image component of this pre-trained model was then fine-tuned
for condition classification and a couple of other downstream tasks using a
number of clinical datasets (see below).

*   Base model (pre-training): A large number of health-related image-text pairs
    from the public web
*   SFT (supervised fine-tuned) model: tele-dermatology datasets from the United
    States and Colombia, a skin cancer dataset from Australia, and additional
    public images. The images come from a mix of device types, including images
    from smartphone cameras, other cameras, and dermatoscopes. The images also
    have a mix of image takers; images may have been taken by clinicians during
    consultations or self-captured by patients.

### Labeling

Labeling sources vary by dataset. Examples include:

*   (image, caption) pairs from the public web
*   Dermatology condition labels provided by dermatologists labelers funded by
    Google
*   Dermatology condition labels provided with a clinical dataset based on a
    telehealth visit, an in-person visit, or a biopsy

## License

The use of Derm Foundation is governed by the
[Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms).

## Implementation information

Details about the model internals.

### Software

Training was done using [JAX](https://github.com/jax-ml/jax)

JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models.

## Use and limitations

### Intended use

*   Derm Foundation can reduce the training data, compute, and technical
    expertise necessary to develop task-specific models for skin image analysis.

*   Embeddings from the model can be used for a variety of user-defined
    downstream tasks including, but not limited to:

    *   Classifying clinical conditions like psoriasis, melanoma or dermatitis
    *   Scoring severity or progression of clinical conditions
    *   Identifying the body part the skin is from
    *   Determining image quality for dermatological assessment

*   To see how to use the model to train a classifier see this
    [Linear classifier example](https://colab.research.google.com/github/google-health/derm-foundation/blob/master/notebooks/train_data_efficient_classifier.ipynb)

### Benefits

*   Derm Foundation Embeddings can be used for efficient training of AI
    development for skin image analysis with significantly less data and compute
    than traditional methods.

*   By leveraging the large set of pre-trained images Derm Foundation is trained
    on, users need less data but can also build more generalizable models than
    training on more limited datasets.

### Limitations

*   Derm Foundation is trained on images with various lightning and noise
    conditions captured in a real-world environment. However, its quality can
    degrade in extreme conditions, such as photos that are too light or too
    dark.

*   The base model was trained using image-text pairs from the public web. These
    images come from a variety of sources but may by noisy or low-quality. The
    SFT (supervised fine-tuned) model was trained data from a limited set of
    countries (United States, Colombia, Australia, public images) and settings
    (mostly clinical). It may not generalize well to data from other countries,
    patient populations, or image types not used in training.

*   The model is only used to generate embeddings of user-provided data. It does
    not generate any predictions or diagnosis on its own.

*   As with any research, developers should ensure any downstream application is
    validated to understand performance using data that is appropriately
    representative of the intended use setting for the specific application
    (e.g., skin tone/type, age, sex, gender etc.).