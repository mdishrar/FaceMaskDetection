Face Mask Detection System
📋 Project Overview

A deep learning-based face mask detection system that can identify whether a person in an image is wearing a mask or not. This project uses Convolutional Neural Networks (CNN) to classify images into two categories: "with mask" and "without mask".
🎯 Features

    Real-time Detection: Can process images to detect mask usage

    High Accuracy: Achieves over 90% accuracy on test data

    Easy to Use: Simple API for making predictions

    Scalable: Can be extended to video streams or real-time camera feeds

🏗️ Project Structure
text

FaceMaskDetection/
├── data/                    # Dataset directory
│   ├── with_mask/          # Images of people wearing masks
│   └── without_mask/       # Images of people without masks
├── models/                  # Saved model files
├── src/                    # Source code
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training script
│   └── predict.py         # Prediction script
├── notebooks/              # Jupyter notebooks for experimentation
├── requirements.txt        # Python dependencies
└── README.md              # This file

🚀 Installation
Prerequisites

    Python 3.8+

    pip (Python package manager)

Step-by-Step Installation

    Clone the repository

bash

git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection

    Create a virtual environment (recommended)

bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies

bash

pip install -r requirements.txt

📊 Dataset Preparation
Dataset Structure

The dataset should be organized as follows:
text

data/
├── with_mask/      # 1629 images
└── without_mask/   # 3828 images

Data Preprocessing

    All images are resized to 128×128 pixels

    Images are converted to RGB format

    Pixel values are normalized to [0, 1] range

    Dataset is split into training (80%) and testing (20%) sets

🧠 Model Architecture

The CNN model consists of:

    Input Layer: 128×128×3 (RGB images)

    Convolutional Layers:

        Conv2D (32 filters, 3×3 kernel, ReLU activation)

        MaxPooling2D (2×2)

        Conv2D (32 filters, 3×3 kernel, ReLU activation)

        MaxPooling2D (2×2)

    Fully Connected Layers:

        Flatten layer

        Dense layer (128 neurons, ReLU activation)

        Dropout (0.5)

        Dense layer (64 neurons, ReLU activation)

        Dropout (0.5)

    Output Layer: Dense layer (2 neurons, softmax activation)

Training Configuration

    Optimizer: Adam

    Loss Function: Sparse Categorical Crossentropy

    Metrics: Accuracy

    Epochs: 5-10

    Batch Size: 32

    Validation Split: 10%

📈 Performance Metrics
Training Results (5 epochs)
text

Epoch 1/5: Train Accuracy: 71.99%, Validation Accuracy: 86.04%
Epoch 2/5: Train Accuracy: 87.44%, Validation Accuracy: 87.41%
Epoch 3/5: Train Accuracy: 90.89%, Validation Accuracy: 89.02%
Epoch 4/5: Train Accuracy: 93.56%, Validation Accuracy: 91.99%
Epoch 5/5: Train Accuracy: 93.83%, Validation Accuracy: 91.76%

Model Performance

    Final Training Accuracy: 93.83%

    Final Validation Accuracy: 91.76%

    Training Loss: 0.17

    Validation Loss: 0.205

💻 Usage
1. Training the Model
python

# Run training script
python src/train.py --data_path /path/to/data --epochs 10

2. Making Predictions
python

# Using the predict script
python src/predict.py --image_path /path/to/image.jpg

3. Interactive Prediction (Colab/Notebook)
python

from src.predict import predict_mask

# Load and preprocess image
result = predict_mask('/path/to/image.jpg', model)
print(f"Prediction: {result}")

4. Real-time Camera Detection
python

# For real-time webcam detection
python src/camera_detection.py

🔧 Handling Class Imbalance

The dataset has inherent class imbalance:

    With mask: 1,629 images

    Without mask: 3,828 images

Solutions Implemented:

    Class Weighting: Automatically assigns higher weights to minority class

    Data Augmentation: Random rotations, shifts, and flips

    Architecture Tuning: Dropout layers to prevent overfitting

📁 File Descriptions
Main Scripts

    train.py: Handles data loading, model training, and saving

    predict.py: Contains functions for making predictions on new images

    model.py: Defines the CNN architecture

    data_processing.py: Data loading and preprocessing utilities

Notebooks

    exploratory_analysis.ipynb: Data exploration and visualization

    model_training.ipynb: Step-by-step model training

    prediction_demo.ipynb: Demonstration of prediction capabilities

🛠️ Dependencies
Core Libraries
text

tensorflow>=2.12.0
keras>=2.12.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
Pillow>=9.5.0

Development Dependencies
text

jupyter>=1.0.0
notebook>=6.5.0
black>=23.0.0
flake8>=6.0.0

Install all dependencies:
bash

pip install -r requirements.txt

🧪 Testing

Run tests to ensure model functionality:
bash

# Run unit tests
python -m pytest tests/

# Test prediction on sample images
python tests/test_predictions.py

🔍 Troubleshooting
Common Issues and Solutions

    Memory Error during training

        Reduce batch size

        Use image generators instead of loading all images at once

    Low accuracy on new images

        Ensure images are preprocessed correctly (128×128, RGB)

        Check if training data is representative

        Consider data augmentation

    Class imbalance issues

        Use the provided class weighting

        Add more data to minority class

        Try different loss functions (focal loss)

📚 API Reference
predict_mask(image_path, model, threshold=0.5)

Predicts whether a person in an image is wearing a mask.

Parameters:

    image_path (str): Path to the image file

    model (keras.Model): Trained CNN model

    threshold (float): Confidence threshold (default: 0.5)

Returns:

    str: Prediction result with confidence score

Example:
python

result = predict_mask('test_image.jpg', model)
print(result)  # Output: "Mask (Confidence: 95.23%)"

🤝 Contributing

    Fork the repository

    Create a feature branch (git checkout -b feature/AmazingFeature)

    Commit your changes (git commit -m 'Add some AmazingFeature')

    Push to the branch (git push origin feature/AmazingFeature)

    Open a Pull Request

Contribution Guidelines

    Follow PEP 8 style guide

    Add tests for new features

    Update documentation accordingly

    Ensure backward compatibility

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    Dataset providers and contributors

    TensorFlow and Keras teams for excellent deep learning frameworks

    OpenCV community for computer vision tools

📞 Contact

For questions, suggestions, or collaborations:

    Email: your.email@example.com

    GitHub Issues: Create an issue

    Pull Requests: Welcome!

📊 Future Improvements

    Real-time video stream processing

    Mobile app integration

    Support for multiple faces in one image

    Integration with temperature sensors

    Cloud deployment and API service

    Edge device optimization (Raspberry Pi, Jetson Nano)

⭐ Star this repository if you found it useful!

Last Updated: February 1, 2026

Dataset = https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
