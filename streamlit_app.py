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
        raise ValueError("Waveform must contain at least two seconds of data.")
    if not np.isfinite(trace).all():
        raise ValueError("Waveform contains NaN or infinite values.")
    return trace


def detect_p_wave(trace, cnn, channel_mean, channel_std):
    starts = np.arange(0, len(trace) - WINDOW_SIZE + 1, STRIDE)
    windows = np.stack([trace[s:s + WINDOW_SIZE] for s in starts]).astype(np.float32)
    normalized = (windows - channel_mean) / (channel_std + 1e-8)
    probabilities = cnn.predict(normalized, batch_size=256, verbose=0).ravel()
    triggered = np.flatnonzero(probabilities >= THRESHOLD)
    if not len(triggered):
        return None, probabilities, starts
    index = int(triggered[0])
    decision_time = (starts[index] + WINDOW_SIZE) / FS
    estimated_p_time = decision_time - MEAN_DETECTION_LATENCY
    return {
        "probability": float(probabilities[index]),
        "decision_time": float(decision_time),
        "estimated_p_time": float(estimated_p_time),
        "p_sample": int(round(estimated_p_time * FS)),
    }, probabilities, starts


def arrival_rules(is_reference, result):
    rules = []
    if is_reference:
        rules.append((REFERENCE_P_TIME, "Actual P-wave", "#16a34a"))
    if result is not None:
        rules.append((result["estimated_p_time"], "Estimated P-wave", "#dc2626"))
    return rules


def add_arrival_rules(chart, rules):
    for arrival_time, label, color in rules:
        marker = alt.Chart(pd.DataFrame({"Time": [arrival_time]})).mark_rule(
            color=color, strokeDash=[6, 4], strokeWidth=2.5
        ).encode(x="Time:Q", tooltip=[alt.Tooltip("Time:Q", format=".2f", title=label)])
        chart = chart + marker
    return chart


def show_waveform_charts(trace, result, probabilities, starts, is_reference):
    st.subheader("Seismic Waveform Analysis")
    st.caption("Green dashed line: actual labelled P-wave. Red dashed line: estimated P-wave.")
    time = np.arange(len(trace)) / FS
    rules = arrival_rules(is_reference, result)

    for channel, label in enumerate(["East (E)", "North (N)", "Vertical (Z)"]):
        data = pd.DataFrame({"Time": time, "Amplitude": trace[:, channel]})
        line = alt.Chart(data).mark_line(strokeWidth=1).encode(
            x=alt.X("Time:Q", title="Time (seconds)", scale=alt.Scale(domain=[0, float(time[-1])])),
            y=alt.Y("Amplitude:Q", title="Amplitude"),
            tooltip=[alt.Tooltip("Time:Q", format=".2f"), alt.Tooltip("Amplitude:Q", format=".3f")],
        ).properties(title=label, height=180)
        st.altair_chart(add_arrival_rules(line, rules), use_container_width=True)

    decision_times = (np.asarray(starts) + WINDOW_SIZE) / FS
    probability_data = pd.DataFrame({"Time": decision_times, "Probability": probabilities})
    prob_line = alt.Chart(probability_data).mark_line(color="#7c3aed").encode(
        x=alt.X("Time:Q", title="Decision time (seconds)"),
        y=alt.Y("Probability:Q", title="CNN probability", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("Time:Q", format=".2f"), alt.Tooltip("Probability:Q", format=".4f")],
    ).properties(title="CNN P-wave Detection Probability", height=220)
    threshold_rule = alt.Chart(pd.DataFrame({"Threshold": [THRESHOLD]})).mark_rule(
        color="#6b7280", strokeDash=[3, 3]
    ).encode(y="Threshold:Q")
    st.altair_chart(add_arrival_rules(prob_line + threshold_rule, rules), use_container_width=True)
    st.caption(f"Detection probability threshold: {THRESHOLD:.2f}")


st.title("🌍 Earthquake Early Warning System")
st.subheader("P-wave Detection and Magnitude Estimation")
st.caption("Machine Learning Research Prototype | 100 Hz | Three-component seismic waveform")
st.info("Upload a CSV or NPY waveform with three components in East (E), North (N), Vertical (Z) order.")

uploaded_file = st.file_uploader("Upload seismic waveform", type=["csv", "npy"])

if uploaded_file is not None and st.button("Analyze Earthquake", type="primary"):
    try:
        with st.spinner("Loading models and analyzing waveform..."):
            cnn, magnitude_model, channel_mean, channel_std, columns = load_models()
            trace = load_waveform(uploaded_file)
            result, probabilities, starts = detect_p_wave(trace, cnn, channel_mean, channel_std)

        st.success("Waveform loaded successfully.")
        is_reference = uploaded_file.name == REFERENCE_FILENAME
        if is_reference:
            st.info(f"Reference case selected: {REFERENCE_ID}")
            st.caption("Reference matching is based on filename only. It does not authenticate file contents.")

        show_waveform_charts(trace, result, probabilities, starts, is_reference)

        if result is None:
            st.warning("No P-wave trigger detected at the selected threshold.")
