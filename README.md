# Human Action Recognition & Face Detection using Python

This project demonstrates practical applications of machine learning and computer vision in Python. It combines **human activity recognition** (from smartphone accelerometer data) with **face detection/recognition** techniques to explore real-world sensing and identification problems.

Two main tracks are explored:
- Classifying everyday human movements (walking, jogging, sitting, etc.) using time-series accelerometer signals.
- Detecting and recognizing human faces in images or video streams.

Built with modern Python tools, this repo serves as a learning showcase and starting point for similar sensor-based or vision-based ML projects.

## Key Features
- **Activity Recognition from Motion Data**  
  Train models to identify user activities or even individual users based on how they walk/move.
- **Face Detection & Recognition**  
  Implement basic to advanced face handling (detection, landmark extraction, simple matching).
- End-to-end pipeline: data preprocessing → feature engineering → model training → evaluation/visualization.
- Clean, modular code with Jupyter notebooks for experimentation.

## Technologies Used
- **Python 3.10+** (developed and tested on 3.12/3.14 compatible)
- Core libraries:
  - TensorFlow (for deep learning models like CNNs or LSTMs on time-series)
  - NumPy (array operations & math)
  - Pandas (data handling & exploration)
  - Matplotlib (visualizing signals, confusion matrices, training curves)

## Datasets

### 1. Human Activity Recognition (Main Focus)
We use the popular **WISDM Actitracker** dataset from the Wireless Sensor Data Mining Lab (Fordham University).  
It contains real-world smartphone accelerometer readings collected from users performing everyday activities.

- **Activities included**: Walking, Jogging, Sitting, Standing, Upstairs, Downstairs (and more in variants)
- **Sampling rate**: ~20 Hz
- **Why this dataset?** It's realistic, unbalanced (reflects real life), and great for benchmarking HAR models.

**Download**:  
[Official WISDM Dataset Page](https://www.cis.fordham.edu/wisdm/dataset.php)  
(Recommended: Grab the latest Actitracker version – raw or transformed files)

### 2. Related / Alternative Dataset (User Identification)
For the "who's walking?" problem (biometric gait recognition):  
**User Identification From Walking Activity** (UCI Machine Learning Repository)  
- Accelerometer data from 22 people walking naturally (phone in chest pocket)  
- Great for experimenting with person authentication via motion patterns.

**Download**:  
[UCI Repository – User Identification From Walking Activity](https://archive.ics.uci.edu/dataset/459/user+identification+from+walking+activity)

## Project Structure (Typical)

Human-Action-And-Face-Recognition-using-Python/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Load, visualize, understand signals
│   ├── 02_preprocessing_feature_eng.ipynb # Windowing, stats, FFT, etc.
│   ├── 03_model_training_activity.ipynb   # CNN / LSTM for activity classification
│   ├── 04_face_detection_recognition.ipynb# OpenCV / dlib / deepface basics
│   └── evaluation_visuals.ipynb
├── src/                  # Reusable scripts & utils
├── data/                 # (gitignore large files – place datasets here)
├── requirements.txt
├── README.md
└── models/               # Saved trained models (optional)


## Quick Start
1. Clone the repo:
   ```bash
   https://github.com/SKarthik12321/Human-Action-And-Face-Recognition-using-Python.git
   cd Human-Action-And-Face-Recognition-using-Python


Install dependencies:Bashpip install -r requirements.txt(Or manually: tensorflow numpy pandas matplotlib + any extras like opencv-python for face part)
Download datasets (links above) and place them in /data/
Open the notebooks in Jupyter / VS Code / Colab and run step-by-step.

Results & Learnings (Example Highlights)

Achieved strong accuracy on WISDM activities with simple CNNs on segmented windows.
Face detection works reliably in real-time with webcam (extendable to recognition).
Key challenges: class imbalance, noisy real-world sensor data, window size tuning.

Feel free to fork, experiment, and improve! Contributions welcome – especially adding more advanced models (Transformers for time-series, lightweight face models).

License
MIT – free to use, modify, and share.
Happy coding & exploring human motion + vision AI! 🚀
text
