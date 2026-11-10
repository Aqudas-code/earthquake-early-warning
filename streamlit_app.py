import io
import json
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

FS = 100
WINDOW_SIZE = 100
STRIDE = 10
THRESHOLD = 0.5
MEAN_DETECTION_LATENCY = 0.3409

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REFERENCE_FILENAME = "reference_earthquake_109C.csv"
REFERENCE_ID = "109C.TA_20070904204109_EV"
REFERENCE_MAGNITUDE = 2.27
REFERENCE_P_TIME = 9.00

st.set_page_config(page_title="Earthquake Early Warning", page_icon="🌍", layout="wide")

@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "pwave_cnn_v2.keras"))
    magnitude = joblib.load(os.path.join(MODEL_DIR, "magnitude_random_forest.pkl"))
    mean = np.load(os.path.join(MODEL_DIR, "channel_mean.npy"))
    std = np.load(os.path.join(MODEL_DIR, "channel_std.npy"))
    with open(os.path.join(BASE_DIR, "feature_columns.json"), encoding="utf-8") as f:
        columns = json.load(f)
    return cnn, magnitude, mean, std, columns


def extract_signal_features(signal, prefix):
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float64))
    rms = np.sqrt(np.mean(signal ** 2))
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1 / FS)
    power = spectrum ** 2
    total_power = np.sum(power) + 1e-12

    def band_ratio(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        return np.sum(power[mask]) / total_power

    return {
        f"{prefix}_rms": rms,
        f"{prefix}_std": np.std(signal),
        f"{prefix}_peak": np.max(np.abs(signal)),
        f"{prefix}_mean_abs": np.mean(np.abs(signal)),
        f"{prefix}_energy": np.sum(signal ** 2),
        f"{prefix}_peak_to_peak": np.ptp(signal),
        f"{prefix}_dominant_freq": frequencies[np.argmax(spectrum)],
        f"{prefix}_spectral_centroid": np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-12),
        f"{prefix}_low_band_ratio": band_ratio(1, 5),
        f"{prefix}_mid_band_ratio": band_ratio(5, 10),
        f"{prefix}_high_band_ratio": band_ratio(10, 30),
    }


def extract_magnitude_features(trace, p_sample, columns):
    if p_sample < WINDOW_SIZE or p_sample + WINDOW_SIZE > len(trace):
        raise ValueError("Not enough waveform samples around the estimated P-wave arrival.")
    p_window = trace[p_sample:p_sample + WINDOW_SIZE]
    noise_window = trace[p_sample - WINDOW_SIZE:p_sample]
    features = {}
    for channel, name in enumerate(["E", "N", "Z"]):
        features.update(extract_signal_features(p_window[:, channel], name))
        p_rms = np.sqrt(np.mean(p_window[:, channel] ** 2))
        noise_rms = np.sqrt(np.mean(noise_window[:, channel] ** 2))
        features[f"{name}_snr"] = p_rms / (noise_rms + 1e-12)
    combined = np.sqrt(np.sum(p_window ** 2, axis=1))
    features.update(extract_signal_features(combined, "combined"))
    return pd.DataFrame([features])[columns]


def load_waveform(uploaded_file):
    content = io.BytesIO(uploaded_file.getvalue())
    if uploaded_file.name.lower().endswith(".npy"):
        trace = np.load(content, allow_pickle=False)
    elif uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(content)
        if all(name in df.columns for name in ["E", "N", "Z"]):
            trace = df[["E", "N", "Z"]].to_numpy()
        else:
            trace = df.select_dtypes(include=[np.number]).to_numpy()
    else:
        raise ValueError("Upload a CSV or NPY file.")
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("Waveform must contain exactly three numeric columns in E, N, Z order.")
    if len(trace) < 2 * WINDOW_SIZE:
