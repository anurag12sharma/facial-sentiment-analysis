# 🎭 Facial Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

<img width="1940" height="1336" alt="image" src="https://github.com/user-attachments/assets/70e8774e-0174-4733-b988-ef1e56292897" />
A comprehensive emotion detection application that analyses sentiment from both facial images and text input using deep learning models. Built with Streamlit for an interactive web interface.

## 🌟 Features

- **🖼️ Facial Emotion Detection**: Upload images to detect emotions from facial expressions
- **📝 Text Sentiment Analysis**: Analyse emotions from text input with confidence scores
- **📊 Interactive Visualizations**: Real-time probability charts for emotion predictions
- **⚡ Real-time Processing**: Instant results with optimized model loading
- **🎨 User-friendly Interface**: Clean, intuitive Streamlit web app

## 🔧 Technical Architecture

### Models Used

#### 1. Facial Emotion Detection
- **Architecture**: ResNet-18 (Transfer Learning)
- **Framework**: PyTorch
- **Input**: 224x224 RGB images
- **Training**: FER dataset with data augmentation
- **Performance**: Optimized for real-time inference

#### 2. Text Sentiment Analysis  
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectorization
- **Framework**: scikit-learn
- **Preprocessing**: Text cleaning and normalization

### Tech Stack
- **Frontend**: Streamlit
- **ML/DL**: PyTorch, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: Altair
- **Image Processing**: PIL, torchvision

## 📁 Project Structure

```
facial-sentiment-analysis/
├── app.py                                  # Main Streamlit application
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore rules
├── Models/
│   ├── FacialModel.py                      # Facial emotion model training script
│   ├── Emotion Detection in Text.ipynb     # Text emotion analysis notebook
│   ├── emotion_model.pth*                  # Trained facial emotion model (not in git)
│   └── text_emotion.pkl*                   # Trained text sentiment model (not in git)
└── Datasets/                               # Training datasets (not in git)
    ├── FERdataset/                         # Facial emotion recognition dataset
    │   ├── train/                          # Training images by emotion
    │   └── test/                           # Test images by emotion
    └── TextDataset/
        └── emotion_dataset_raw.csv         # Text emotion dataset
```

## 🚀 Screenshots
> <img width="1940" height="1336" alt="pawelzmarlak-2025-09-19T20_26_25 158Z" src="https://github.com/user-attachments/assets/d5ee778f-0823-4f7f-8bd6-cf92bbab2cc2" />
> <img width="1940" height="1126" alt="pawelzmarlak-2025-09-19T20_26_59 102Z" src="https://github.com/user-attachments/assets/5b2ff77b-e103-44f4-aa2b-4e291188f28f" />
> <img width="1940" height="1068" alt="pawelzmarlak-2025-09-19T20_27_18 094Z" src="https://github.com/user-attachments/assets/8eccda84-b518-4a18-aaba-7880793bf452" />




## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anurag12sharma/facial-sentiment-analysis.git
   cd facial-sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models**
   
   Due to GitHub's file size limitations, you'll need to download the trained models separately:
   
   - Create a `Models` directory if it doesn't exist
   - Download the models from [Google Drive](https://drive.google.com/drive/folders/16z_y3v8UQf_RFpOL5pnHt-nsynY8yn61?usp=sharing)
   - Place the following files in the `Models/` directory:
     - `emotion_model.pth` (Facial emotion detection model - 43MB)
     - `text_emotion.pkl` (Text sentiment analysis model - 12MB)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`


## 🎯 Usage Examples

### Image Emotion Detection
1. Select "Image" option
2the . Upload a facial image (JPG, PNG, JPEG)
3. View the predicted emotion instantly

### Text Sentiment Analysis
1. Select "Text" option  
2.the  Enter your text in the text area
3. Click "Submit" to see:
   - Predicted emotion with emoji
   - Confidence score
   - Probability distribution chart

## 🔬 Model Performance

### Facial Emotion Model
- **Architecture**: ResNet-18 with custom classifier
- **Training Epochs**: 6
- **Optimizer**: Adam (lr=0.001)
- **Data Augmentation**: Grayscale conversion, normalization

### Text Sentiment Model
- **Algorithm**: Logistic Regression with TF-IDF
- **Features**: Cleaned text with noise removal
- **Validation**: Cross-validation for robust performance


## 📊 Dataset Information

- **Facial Emotions**: FER (Facial Expression Recognition) dataset
- **Text Emotions**: Custom curated emotion dataset
- **Training Split**: 80% train, 20% test
- **Data Preprocessing**: Image normalization, text cleaning


## 👨‍💻 Author

**Anurag Sharma**
- GitHub: [@anurag12sharma](https://github.com/anurag12sharma)
- LinkedIn: [Connect with me](https://linkedin.com/in/anurag12sharma)

## 🙏 Acknowledgments

- ResNet architecture by Microsoft Research
- FER dataset contributors
- Streamlit team for the amazing framework
- PyTorch and scikit-learn communities

---

⭐ **Star this repository if you found it helpful!**
