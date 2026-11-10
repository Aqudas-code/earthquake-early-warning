
import os
import io
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

FS = 100
WINDOW_SIZE = 100
STRIDE = 10
THRESHOLD = 0.5

# Approximate average detection latency from the research experiment
MEAN_DETECTION_LATENCY = 0.3409

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# STEAD reference recording
REFERENCE_FILENAME = "reference_earthquake_109C.csv"
REFERENCE_ID = "109C.TA_20070904204109_EV"
REFERENCE_MAGNITUDE = 2.27
REFERENCE_P_TIME = 9.00

st.set_page_config(
    page_title="Earthquake Early Warning",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    cnn = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "pwave_cnn_v2.keras")
    )

    magnitude_model = joblib.load(
        os.path.join(MODEL_DIR, "magnitude_random_forest.pkl")
    )

    channel_mean = np.load(
        os.path.join(MODEL_DIR, "channel_mean.npy")
    )

    channel_std = np.load(
        os.path.join(MODEL_DIR, "channel_std.npy")
    )

    with open(
        os.path.join(BASE_DIR, "feature_columns.json"),
        "r"
    ) as f:
        feature_columns = json.load(f)

    return (
        cnn,
        magnitude_model,
        channel_mean,
        channel_std,
        feature_columns
    )


# ============================================================
# 3. SIGNAL FEATURE EXTRACTION
# ============================================================

def extract_signal_features(signal, prefix):

    signal = np.asarray(signal, dtype=np.float64)
    signal = np.nan_to_num(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    energy = np.sum(signal ** 2)

    spectrum = np.abs(np.fft.rfft(signal))

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / FS
    )

    total_power = np.sum(spectrum ** 2) + 1e-12

    dominant_frequency = frequencies[
        np.argmax(spectrum)
    ]

    centroid = (
        np.sum(frequencies * spectrum) /
        (np.sum(spectrum) + 1e-12)
    )

    def band_ratio(low, high):

        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )

        return (
            np.sum(spectrum[mask] ** 2) /
            total_power
        )

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


# ============================================================
# 4. MAGNITUDE FEATURE EXTRACTION
# ============================================================

def extract_magnitude_features(
    trace,
    p_sample,
    feature_columns
):

    p_start = int(p_sample)
    p_end = p_start + WINDOW_SIZE

    noise_start = p_start - WINDOW_SIZE
    noise_end = p_start

    if noise_start < 0 or p_end > len(trace):
        raise ValueError(
            "Not enough waveform samples around the estimated "
            "P-wave arrival to calculate magnitude features."
        )

    p_window = trace[p_start:p_end]
    noise_window = trace[noise_start:noise_end]

    features = {}

    for channel, name in enumerate(["E", "N", "Z"]):

        features.update(
            extract_signal_features(
                p_window[:, channel],
                name
            )
        )

        p_rms = np.sqrt(
            np.mean(p_window[:, channel] ** 2)
        )

        noise_rms = np.sqrt(
            np.mean(noise_window[:, channel] ** 2)
        )

        features[f"{name}_snr"] = (
            p_rms / (noise_rms + 1e-12)
        )

    combined = np.sqrt(
        np.sum(p_window ** 2, axis=1)
    )

    features.update(
        extract_signal_features(
            combined,
            "combined"
        )
    )

    feature_df = pd.DataFrame([features])

    return feature_df[feature_columns]


# ============================================================
# 5. LOAD WAVEFORM
# ============================================================

def load_waveform(uploaded_file):

    extension = uploaded_file.name.lower().split(".")[-1]

    if extension == "npy":

        trace = np.load(
            io.BytesIO(uploaded_file.getvalue()),
            allow_pickle=False
        )

    elif extension == "csv":

        dataframe = pd.read_csv(
            io.BytesIO(uploaded_file.getvalue())
        )

        trace = dataframe.select_dtypes(
            include=[np.number]
        ).to_numpy()

    else:

        raise ValueError(
            "Please upload a CSV or NPY waveform."
        )

    trace = np.asarray(
        trace,
        dtype=np.float32
    )

    if trace.ndim != 2 or trace.shape[1] != 3:

        raise ValueError(
            "Waveform must contain exactly three numeric columns: E, N, Z."
        )

    if len(trace) < 2 * WINDOW_SIZE:

        raise ValueError(
            "Waveform must contain at least two seconds of data."
        )

    return np.nan_to_num(trace)


# ============================================================
# 6. CNN P-WAVE DETECTION
# ============================================================

def detect_p_wave(
    trace,
    cnn,
    channel_mean,
    channel_std
):

    starts = list(
        range(
            0,
            len(trace) - WINDOW_SIZE + 1,
            STRIDE
        )
    )

    windows = np.asarray(
        [
            trace[start:start + WINDOW_SIZE]
            for start in starts
        ],
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

    triggered = np.where(
        probabilities >= THRESHOLD
    )[0]

    if len(triggered) == 0:

        return None, probabilities, starts

    index = int(triggered[0])

    decision_time = (
        starts[index] + WINDOW_SIZE
    ) / FS

    estimated_p_time = (
        decision_time - MEAN_DETECTION_LATENCY
    )

    p_sample = int(
        round(estimated_p_time * FS)
    )

    result = {
        "probability": float(probabilities[index]),
        "decision_time": float(decision_time),
        "estimated_p_time": float(estimated_p_time),
        "p_sample": p_sample
    }

    return result, probabilities, starts


# ============================================================
# 7. DASHBOARD HEADER
# ============================================================

st.title("🌍 Earthquake Early Warning System")

st.subheader(
    "P-wave Detection and Magnitude Estimation"
)

st.caption(
    "Machine Learning Research Prototype | "
    "100 Hz | Three-component seismic waveform"
)

st.info(
    "Upload a seismic waveform containing three numeric "
    "components in this order: East (E), North (N), Vertical (Z)."
)


# ============================================================
# 8. FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload seismic waveform",
    type=["csv", "npy"]
)


# ============================================================
# 9. ANALYZE WAVEFORM
# ============================================================

if uploaded_file is not None:

    if st.button(
        "Analyze Earthquake",
        type="primary"
    ):

        try:

            with st.spinner(
                "Loading models and analyzing waveform..."
            ):

                (
                    cnn,
                    magnitude_model,
                    channel_mean,
                    channel_std,
                    feature_columns
                ) = load_models()

                trace = load_waveform(
                    uploaded_file
                )

                result, probabilities, starts = detect_p_wave(
                    trace,
                    cnn,
                    channel_mean,
                    channel_std
                )

            st.success(
                "Waveform loaded successfully!"
            )

            # Reference selection by filename
            is_reference = (
                uploaded_file.name == REFERENCE_FILENAME
            )

            if is_reference:

                st.info(
                    f"STEAD reference recording: {REFERENCE_ID}"
                )

            # ====================================================
            # 10. CALCULATE MAGNITUDE
            # ====================================================

            magnitude = None
            lower_spread = None
            upper_spread = None

            if result is not None:

                features = extract_magnitude_features(
                    trace,
                    result["p_sample"],
                    feature_columns
                )

                magnitude = float(
                    magnitude_model.predict(features)[0]
                )

                tree_predictions = np.array([
                    tree.predict(features)[0]
                    for tree in magnitude_model.estimators_
                ])

                lower_spread = float(
                    np.percentile(
                        tree_predictions,
                        5
                    )
                )

                upper_spread = float(
                    np.percentile(
                        tree_predictions,
                        95
                    )
                )

            
            # ====================================================
            # 11. SEISMIC WAVEFORM GRAPHS
            # ====================================================

            st.divider()
            st.subheader("Seismic Waveform Analysis")

            time_axis = np.arange(len(trace)) / FS

            # Prepare waveform data
            waveform_df = pd.DataFrame({
                "Time (seconds)": time_axis,
                "East (E)": trace[:, 0],
                "North (N)": trace[:, 1],
                "Vertical (Z)": trace[:, 2]
            })

            st.markdown("### Three-component seismic waveform")

            st.line_chart(
                waveform_df.set_index("Time (seconds)"),
                height=280,
                use_container_width=True
            )

            # Display reference and estimated arrival times
            if is_reference:
                st.info(
                    f"Actual P-wave arrival: {REFERENCE_P_TIME:.2f} seconds"
                )

            if result is not None:
                st.info(
                    f"Estimated P-wave arrival: "
                    f"{result['estimated_p_time']:.2f} seconds"
                )

            # CNN probability graph
            probability_times = (
                np.asarray(starts) + WINDOW_SIZE
            ) / FS

            probability_df = pd.DataFrame({
                "Decision time (seconds)": probability_times,
                "CNN probability": probabilities
            })

            st.markdown("### CNN P-wave detection probability")

            st.line_chart(
                probability_df.set_index("Decision time (seconds)"),
                height=250,
                use_container_width=True
            )

            st.caption(
                f"Detection threshold: {THRESHOLD:.2f}"
            )

            # CNN probability graph

            probability_times = (
                np.asarray(starts) + WINDOW_SIZE
            ) / FS

            axes[3].plot(
                probability_times,
                probabilities,
                color="purple",
                linewidth=1,
                label="CNN Detection Probability"
            )

            axes[3].axhline(
                THRESHOLD,
                color="black",
                linestyle="--",
                label="Detection Threshold"
            )

            if is_reference:

                axes[3].axvline(
                    REFERENCE_P_TIME,
                    color="green",
                    linestyle="--",
                    label="Actual P-wave arrival"
                )

            if result is not None:

                axes[3].axvline(
                    result["decision_time"],
                    color="red",
                    linestyle=":",
                    label="Detection decision time"
                )

            axes[3].set_title(
                "CNN P-wave Detection Probability"
            )

            axes[3].set_xlabel(
                "Time (seconds)"
            )

            axes[3].set_ylabel(
                "Probability"
            )

            axes[3].legend(
                loc="upper right"
            )

            axes[3].grid(
                True,
                alpha=0.3
            )

            fig.tight_layout()

            st.write(
                "### Seismic Waveform Graph"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            # ====================================================
            # 12. DETECTION RESULTS
            # ====================================================

            if result is None:

                st.warning(
                    "No P-wave trigger detected at the selected threshold."
                )

                st.metric(
                    "Maximum CNN detection probability",
                    f"{float(np.max(probabilities)):.4f}"
                )

            else:

                st.divider()

                st.subheader(
                    "Earthquake Analysis Results"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "P-wave Detection Probability",
                    f"{result['probability']:.4f}"
                )

                col2.metric(
                    "Estimated Magnitude (ML)",
                    f"{magnitude:.2f}"
                )

                col3, col4 = st.columns(2)

                col3.metric(
                    "Detection Decision Time",
                    f"{result['decision_time']:.2f} s"
                )

                col4.metric(
                    "Estimated P-wave Arrival",
                    f"{result['estimated_p_time']:.2f} s"
                )

                st.write(
                    "**Random Forest tree prediction spread:** "
                    f"{lower_spread:.2f} – {upper_spread:.2f}"
                )

                # ====================================================
                # 13. REFERENCE COMPARISON
                # ====================================================

                if is_reference:

                    st.divider()

                    st.subheader(
                        "Ground Truth vs Model Prediction"
                    )

                    st.caption(
                        "Reference labels are taken from the matched "
                        "STEAD metadata record."
                    )

                    st.write(
                        "**Dataset:** STEAD"
                    )

                    st.write(
                        f"**Recording ID:** `{REFERENCE_ID}`"
                    )

                    st.write(
                        "**Magnitude type:** ML"
                    )

                    arrival_error = abs(
                        result["estimated_p_time"] -
                        REFERENCE_P_TIME
                    )

                    magnitude_error = abs(
                        magnitude -
                        REFERENCE_MAGNITUDE
                    )

                    # Arrival comparison

                    st.markdown(
                        "#### P-wave Arrival Comparison"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Actual P-wave Arrival",
                        f"{REFERENCE_P_TIME:.2f} s"
                    )

                    col2.metric(
                        "Predicted P-wave Arrival",
                        f"{result['estimated_p_time']:.2f} s",
                        delta=(
                            f"{result['estimated_p_time'] - REFERENCE_P_TIME:+.2f} s"
                        )
                    )

                    # Magnitude comparison

                    st.markdown(
                        "#### Magnitude Comparison"
                    )

                    col3, col4 = st.columns(2)

                    col3.metric(
                        "Actual Magnitude (ML)",
                        f"{REFERENCE_MAGNITUDE:.2f}"
                    )

                    col4.metric(
                        "Predicted Magnitude (ML)",
                        f"{magnitude:.2f}",
                        delta=(
                            f"{magnitude - REFERENCE_MAGNITUDE:+.2f}"
                        )
                    )

                    # Errors

                    st.markdown(
                        "#### Absolute Prediction Errors"
                    )

                    err1, err2 = st.columns(2)

                    err1.metric(
                        "P-wave Arrival Error",
                        f"{arrival_error:.2f} seconds"
                    )

                    err2.metric(
                        "Magnitude Error",
                        f"{magnitude_error:.2f} ML"
                    )

                    # Magnitude comparison graph

                    st.markdown(
                        "#### Actual vs Predicted Magnitude Graph"
                    )

                    comparison_fig, ax = plt.subplots(
                        figsize=(8, 4)
                    )

                    labels = [
                        "Actual Magnitude",
                        "Predicted Magnitude"
                    ]

                    values = [
                        REFERENCE_MAGNITUDE,
                        magnitude
                    ]

                    bars = ax.bar(
                        labels,
                        values
                    )

                    ax.set_ylabel(
                        "Magnitude (ML)"
                    )

                    ax.set_title(
                        "Actual vs Predicted Magnitude"
                    )

                    for bar, value in zip(bars, values):

                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{value:.2f}",
                            ha="center",
                            va="bottom"
                        )

                    ax.grid(
                        axis="y",
                        alpha=0.3
                    )

                    comparison_fig.tight_layout()

                    st.pyplot(
                        comparison_fig,
                        use_container_width=True
                    )

                    plt.close(comparison_fig)

                    st.caption(
                        "Reference values are displayed for the selected "
                        "reference filename. Confirm the recording identity "
                        "before using these values in formal evaluation."
                    )

                # ====================================================
                # 14. RESEARCH LIMITATIONS
                # ====================================================

                st.warning(
                    "Research limitations: P-wave arrival uses a fixed "
                    "average latency correction. Magnitude estimation "
                    "depends on that estimated arrival. The tree prediction "
                    "spread is not a calibrated confidence interval. "
                    "This is not a certified public-safety warning system."
                )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "Research Project: P-Wave-Based Earthquake Early Warning "
    "and Magnitude Estimation Using Machine Learning."
)
