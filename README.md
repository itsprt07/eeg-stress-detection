---
title: EEG Stress Detection App
emoji: 🧠
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.46.0
app_file: app.py
pinned: false
---


# eeg-stress-detector-
A Streamlit-based deep learning app for detecting stress from EEG signals

# 🧠 EEG Stress Detection App

This project uses deep learning to detect **stress levels** from EEG (electroencephalogram) signals stored in `.mat` files. It provides a user-friendly web interface to upload EEG data and get real-time stress predictions.

---

## 📚 Dataset Overview

The EEG dataset consists of recordings from **40 human subjects** (14 females, 26 males, mean age: 21.5 years) who were exposed to a series of cognitive and emotional tasks:

- 🧠 **Stroop Color-Word Test**
- ➕ **Arithmetic Problem Solving**
- 🪞 **Symmetric Mirror Image Identification**
- 🧘 **Relaxation State**

These tasks were designed to induce **short-term psychological stress**, allowing for the development of a robust stress prediction model.

---

## 🚀 Features

- ✅ Upload `.mat` EEG files from any user/device
- 📈 Real-time **stress level prediction**
- 🎯 Shows a **stress score** (0 to 1) using an animated slider
- 📊 Displays a **visual preview** of the EEG waveform
- 💻 Built using **TensorFlow**, **Streamlit**, and **Sklearn**

---

## 🧪 Model Architecture

The deep learning model used is an **LSTM (Long Short-Term Memory)** neural network. Key components:

- `LSTM (64)` layers with dropout for temporal learning
- Binary classification using sigmoid activation
- Trained on preprocessed EEG segments
- Achieves strong accuracy for distinguishing between "Stressed" and "Not Stressed"

---

## 🧠 How to Use

1. **Clone the repo:**

   ```bash
   git clone https://github.com/your-username/eeg-stress-detection.git
   cd eeg-stress-detection
2. **Install requirements:**

pip install -r requirements.txt

3. **Run the app:**
   
streamlit run app.py

4. **Upload EEG data (.mat format):**

   File should contain a key like 'Data' with EEG shape e.g., (32, 3200)

5. **View Results:**

   Get a prediction (Stressed or Not Stressed)

   Check stress score and waveform preview

   ## 🧠 Model Training Notebook

You can view the full training & evaluation code in the notebook here:  
👉 [EEG_model_training.ipynb](EEG_model_training.ipynb)


