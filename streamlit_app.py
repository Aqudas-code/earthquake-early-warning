
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

FS = 100
WINDOW_SIZE = 100
STRIDE = 10
THRESHOLD = 0.5
MEAN_DETECTION_LATENCY = 0.3409

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Earthquake Early Warning",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "pwave_cnn_v2.keras")
    )
    magnitude = joblib.load(
        os.path.join(MODEL_DIR, "magnitude_random_forest.pkl")
    )
    mean = np.load(os.path.join(MODEL_DIR, "channel_mean.npy"))
    std = np.load(os.path.join(MODEL_DIR, "channel_std.npy"))

    with open(os.path.join(BASE_DIR, "feature_columns.json")) as f:
        columns = json.load(f)

    return cnn, magnitude, mean, std, columns


def extract_signal_features(signal, prefix):
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.nan_to_num(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    energy = np.sum(signal ** 2)

    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1/FS)

    total_power = np.sum(spectrum ** 2) + 1e-12
    dominant_frequency = frequencies[np.argmax(spectrum)]
    centroid = np.sum(frequencies * spectrum) / (
        np.sum(spectrum) + 1e-12
    )

    def band_ratio(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        return np.sum(spectrum[mask] ** 2) / total_power

    return {
        f"{prefix}_rms": rms,
        f"{prefix}_std": np.std(signal),
        f"{prefix}_peak": peak,
        f"{prefix}_mean_abs": np.mean(np.abs(signal)),
        f"{prefix}_energy": energy,
        f"{prefix}_peak_to_peak": np.ptp(signal),
        f"{prefix}_dominant_freq": dominant_frequency,
        f"{prefix}_spectral_centroid": centroid,
        f"{prefix}_low_band_ratio": band_ratio(1, 5),
        f"{prefix}_mid_band_ratio": band_ratio(5, 10),
        f"{prefix}_high_band_ratio": band_ratio(10, 30)
    }


def extract_magnitude_features(trace, p_sample, feature_columns):
    p_window = trace[p_sample:p_sample + WINDOW_SIZE]
    noise_window = trace[p_sample-WINDOW_SIZE:p_sample]

    if len(p_window) != WINDOW_SIZE or len(noise_window) != WINDOW_SIZE:
        raise ValueError(
            "Not enough samples around the estimated P-wave arrival."
        )

    features = {}

    for channel, name in enumerate(["E", "N", "Z"]):
        features.update(
            extract_signal_features(p_window[:, channel], name)
        )

        p_rms = np.sqrt(np.mean(p_window[:, channel] ** 2))
        noise_rms = np.sqrt(np.mean(noise_window[:, channel] ** 2))

        features[f"{name}_snr"] = p_rms / (noise_rms + 1e-12)

    combined = np.sqrt(np.sum(p_window ** 2, axis=1))
    features.update(extract_signal_features(combined, "combined"))

    return pd.DataFrame([features])[feature_columns]


def load_waveform(uploaded_file):
    extension = uploaded_file.name.lower().split(".")[-1]

    if extension == "npy":
        trace = np.load(
            io.BytesIO(uploaded_file.getvalue()),
            allow_pickle=False
        )
    elif extension == "csv":
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        trace = df.select_dtypes(include=[np.number]).to_numpy()
    else:
        raise ValueError("Upload a CSV or NPY file.")

    trace = np.asarray(trace, dtype=np.float32)

    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError(
            "The waveform must have exactly three numeric columns: E, N, Z."
        )

    if len(trace) < 2 * WINDOW_SIZE:
        raise ValueError("Waveform must contain at least two seconds of data.")

    return np.nan_to_num(trace)


def detect_p_wave(trace, cnn, channel_mean, channel_std):
    starts = list(range(0, len(trace) - WINDOW_SIZE + 1, STRIDE))

    windows = np.asarray(
        [trace[s:s + WINDOW_SIZE] for s in starts],
        dtype=np.float32
    )

    normalized = (
        windows - channel_mean
    ) / (channel_std + 1e-8)

    probabilities = cnn.predict(
        normalized,
        batch_size=256,
        verbose=0
    ).flatten()

    triggered = np.where(probabilities >= THRESHOLD)[0]

    if len(triggered) == 0:
        return None, probabilities, starts

    idx = int(triggered[0])
    decision_time = (starts[idx] + WINDOW_SIZE) / FS
    estimated_p_time = decision_time - MEAN_DETECTION_LATENCY
    p_sample = int(round(estimated_p_time * FS))

    return {
        "probability": float(probabilities[idx]),
        "decision_time": decision_time,
        "estimated_p_time": estimated_p_time,
        "p_sample": p_sample
    }, probabilities, starts


# ---------------- USER INTERFACE ----------------

st.title("🌍 Earthquake Early Warning System")
st.subheader("P-wave Detection and Magnitude Estimation")
st.caption("Machine Learning Research Prototype | 100 Hz | Three-component waveform")

st.info(
    "Upload a seismic waveform containing three numeric components "
    "in this order: East (E), North (N), Vertical (Z)."
)

uploaded_file = st.file_uploader(
    "Upload waveform",
    type=["csv", "npy"]
)

if uploaded_file is not None:
    if st.button("Analyze Earthquake", type="primary"):
        try:
            with st.spinner("Loading models and analyzing waveform..."):
                cnn, magnitude_model, channel_mean, channel_std, columns = (
                    load_models()
                )

                trace = load_waveform(uploaded_file)

                result, probabilities, starts = detect_p_wave(
                    trace, cnn, channel_mean, channel_std
                )

            st.success("Waveform loaded successfully.")

            fig, axes = plt.subplots(2, 1, figsize=(11, 7))

            time_axis = np.arange(len(trace)) / FS

            axes[0].plot(time_axis, trace[:, 2], linewidth=0.6)
            axes[0].set_title("Vertical Component (Z)")
            axes[0].set_xlabel("Time (seconds)")
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(True)

            probability_times = (
                np.asarray(starts) + WINDOW_SIZE
            ) / FS

            axes[1].plot(probability_times, probabilities)
            axes[1].axhline(
                THRESHOLD,
                linestyle="--",
                label="Detection threshold"
            )
            axes[1].set_title("CNN P-wave Detection Probability")
            axes[1].set_xlabel("Decision time (seconds)")
            axes[1].set_ylabel("Probability")
            axes[1].legend()
            axes[1].grid(True)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if result is None:
                st.warning(
                    "No P-wave trigger detected at the selected threshold."
                )
            else:
                p_sample = result["p_sample"]

                features = extract_magnitude_features(
                    trace, p_sample, columns
                )

                magnitude = float(
                    magnitude_model.predict(features)[0]
                )

                tree_predictions = np.array([
                    tree.predict(features)[0]
                    for tree in magnitude_model.estimators_
                ])

                lower_spread = float(
                    np.percentile(tree_predictions, 5)
                )
                upper_spread = float(
                    np.percentile(tree_predictions, 95)
                )

                st.subheader("Earthquake Analysis Results")

                c1, c2 = st.columns(2)

                c1.metric(
                    "P-wave detection probability",
                    f"{result['probability']:.4f}"
                )
                c2.metric(
                    "Estimated magnitude (ML)",
                    f"{magnitude:.2f}"
                )

                c3, c4 = st.columns(2)

                c3.metric(
                    "Detection decision time",
                    f"{result['decision_time']:.2f} s"
                )
                c4.metric(
                    "Estimated P-wave arrival",
                    f"{result['estimated_p_time']:.2f} s"
                )

                st.write(
                    "**Random Forest tree prediction spread:** "
                    f"{lower_spread:.2f} – {upper_spread:.2f}"
                )

                st.warning(
                    "Research limitations: P-wave arrival uses a fixed "
                    "average latency correction. Magnitude estimation "
                    "depends on that estimated arrival. The tree prediction "
                    "spread is not a calibrated confidence interval. "
                    "This is not a certified public-safety warning system."
                )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

st.divider()
st.caption(
    "Research project: P-Wave-Based Earthquake Early Warning "
    "and Magnitude Estimation Using Machine Learning."
)
