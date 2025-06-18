import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="🧠 EEG Stress Detection", layout="centered")
st.markdown(
    """
    <style>
    @media screen and (max-width: 768px) {
        .title {
            font-size: 28px !important;
        }
        .sub {
            font-size: 16px !important;
        }
        .score-label {
            font-size: 18px !important;
        }
        .element-container .stButton>button {
            font-size: 14px !important;
            padding: 8px 12px;
        }
    }

    @media screen and (max-width: 480px) {
        .title {
            font-size: 24px !important;
        }
        .sub {
            font-size: 14px !important;
        }
        .score-label {
            font-size: 16px !important;
        }
    }

    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 20px;
    }
    .sub {
        font-size: 20px;
        text-align: center;
        color: #ccc;
        margin-bottom: 30px;
    }
    .score-label {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="title">🧠 EEG Stress Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload a <code>.mat</code> EEG file to estimate your stress level visually and numerically.</div>', unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("my_model.h5")
expected_shape = model.input_shape
st.info(f"📐 Model expects input shape: {expected_shape}")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your EEG .mat file", type=["mat"])

# Use default test file if no file uploaded
if uploaded_file is None:
    st.info("ℹ️ No file uploaded. Using sample test file for demo.")
    uploaded_file = open("sample_mat_files/Relax_sub_1_trial1.mat", "rb")

try:
    mat_data = scipy.io.loadmat(uploaded_file)
    eeg = mat_data['Data']
    eeg = eeg.T  # shape: (3200, 32)

    # Normalize
    scaler = StandardScaler()
    eeg_scaled = scaler.fit_transform(eeg)

    # EEG waveform preview
    st.subheader("📈 EEG Waveform Preview (First 5 Channels)")
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    for i in range(5):
        axs[i].plot(eeg_scaled[:, i], color='mediumblue')
        axs[i].set_ylabel(f'Ch {i+1}')
        axs[i].grid(True)
    axs[-1].set_xlabel('Time Steps')
    fig.tight_layout()
    st.pyplot(fig)

    # Segment EEG into 32-point windows
    segment_length = 32
    segments = []
    for i in range(0, eeg_scaled.shape[0] - segment_length + 1, segment_length):
        segment = eeg_scaled[i:i+segment_length]
        segments.append(segment)
    segments = np.array(segments)  # (num_segments, 32, 32)

    # Reshape for LSTM: (samples, timesteps, 1)
    segments = segments.reshape(segments.shape[0], segment_length, eeg.shape[1], 1)
    segments = np.mean(segments, axis=2)  # (num_segments, 32, 1)

    st.success(f"✅ Processed {segments.shape[0]} EEG segments.")

    # Simulate prediction delay
    with st.spinner("🧠 Estimating your stress level..."):
        predictions = model.predict(segments)
        avg_pred = np.mean(predictions)

    # Output
    label = "Stressed 😟" if avg_pred > 0.5 else "Not Stressed 😌"

    st.markdown('<div class="score-label">🧪 Estimated Stress Score</div>', unsafe_allow_html=True)
    st.slider("", 0.0, 1.0, float(avg_pred), disabled=True)

    st.markdown(
        f"""
        <div class="score-label">
            🎯 Prediction Result: {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

except Exception as e:
    st.error(f"❌ Error processing file: `{e}`")
# trigger streamlit rebuild
