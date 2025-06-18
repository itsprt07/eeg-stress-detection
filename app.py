import gradio as gr
import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model("my_model.h5")

def predict_stress(mat_file):
    try:
        mat_data = scipy.io.loadmat(mat_file.name)
        eeg = mat_data['Data']
        eeg = eeg.T  # shape (3200, 32)

        # Normalize
        scaler = StandardScaler()
        eeg_scaled = scaler.fit_transform(eeg)

        # Segment EEG into 32-point windows
        segment_length = 32
        segments = []
        for i in range(0, eeg_scaled.shape[0] - segment_length + 1, segment_length):
            segment = eeg_scaled[i:i+segment_length]
            segments.append(segment)
        segments = np.array(segments)

        # Reshape for LSTM input
        segments = segments.reshape(segments.shape[0], segment_length, eeg.shape[1], 1)
        segments = np.mean(segments, axis=2)  # (num_segments, 32, 1)

        predictions = model.predict(segments)
        avg_pred = float(np.mean(predictions))
        label = "Stressed 😟" if avg_pred > 0.5 else "Not Stressed 😌"
        return f"Predicted Score: {avg_pred:.4f}\n\nResult: {label}"
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_stress,
    inputs=gr.File(label="Upload .mat EEG File"),
    outputs="text",
    title="🧠 EEG Stress Detection",
    description="Upload an EEG .mat file and the model will predict whether you're stressed or not.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
