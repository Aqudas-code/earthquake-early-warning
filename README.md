# Earthquake Early Warning and Magnitude Estimation

A software-only machine learning research prototype for P-wave detection and local magnitude (ML) estimation using seismic waveform data.

## Project Overview

The system uses:

1. **CNN (1D Convolutional Neural Network):** Detects potential P-wave arrivals from three-component seismic waveforms.
2. **Random Forest Regressor:** Estimates local magnitude (ML) using 47 waveform features.
3. **Gradio:** Provides an interactive dashboard for waveform analysis.

## Input Format

Upload a CSV or NPY file containing three waveform components:

- E: East
- N: North
- Z: Vertical

Sampling frequency: 100 Hz.

For CSV files, the application uses the numeric columns in E, N, Z order.

## Model Files

- `models/pwave_cnn_v2.keras`
- `models/magnitude_random_forest.pkl`
- `models/channel_mean.npy`
- `models/channel_std.npy`

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Methodology

- Three-component seismic waveform analysis.
- One-second sliding windows with a 0.1-second stride.
- CNN-based P-wave detection.
- Extraction of 47 time-domain and frequency-domain features.
- Random Forest-based local magnitude estimation.

## Important Research Limitations

This is an experimental research prototype.

- The system detects a possible P-wave after an earthquake has begun; it does not predict earthquakes days or weeks in advance.
- Estimated P-wave arrival uses a fixed average detection-latency correction.
- Magnitude estimation depends on the estimated P-wave arrival.
- The Random Forest tree prediction spread is not a calibrated confidence interval.
- The system has not been validated for operational public-safety warnings.

## Technology Stack

Python, TensorFlow, Scikit-learn, NumPy, Pandas, Matplotlib and Gradio.

## Author

Aqudas Rahman
