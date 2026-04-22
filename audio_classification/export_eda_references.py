from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.io import wavfile


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_ROOT = Path(".")
REF_DIR = DATA_ROOT / "references"
REF_DIR.mkdir(exist_ok=True)

MACHINE_TYPES = ["fan", "pump", "slider"]
CHANNEL_TO_ANALYZE = 0

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12

PALETTE = {"normal": "#4C78A8", "abnormal": "#E45756"}


def scan_dataset(machine_types):
    rows = []
    for machine_type in machine_types:
        machine_dir = DATA_ROOT / machine_type
        if not machine_dir.exists():
            continue

        for wav_path in sorted(machine_dir.rglob("*.wav")):
            rel = wav_path.relative_to(DATA_ROOT)
            parts = rel.parts
            if len(parts) < 4:
                continue

            sample_rate, data = wavfile.read(wav_path, mmap=True)
            channels = 1 if data.ndim == 1 else data.shape[1]
            sample_width_bits = data.dtype.itemsize * 8
            n_frames = data.shape[0]
            duration_seconds = n_frames / sample_rate

            rows.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "machine_type": parts[0],
                    "machine_id": parts[1],
                    "status": parts[2],
                    "filename": parts[3],
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width_bits": sample_width_bits,
                    "frames": int(n_frames),
                    "duration_seconds": float(duration_seconds),
                    "dtype": str(data.dtype),
                }
            )

    return pd.DataFrame(rows)


def load_first_channel(path):
    sample_rate, data = wavfile.read(path)
    x = data[:, CHANNEL_TO_ANALYZE] if data.ndim > 1 else data
    x = x.astype(np.float32) / np.iinfo(np.int16).max
    return sample_rate, x


def spectral_features(x, sample_rate):
    freqs, psd = signal.welch(x, fs=sample_rate, nperseg=4096)
    psd = np.maximum(psd, 1e-12)
    psd_sum = psd.sum()

    centroid = float((freqs * psd).sum() / psd_sum)
    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * psd).sum() / psd_sum))
    rolloff_idx = np.searchsorted(np.cumsum(psd), 0.95 * np.cumsum(psd)[-1])
    rolloff_95 = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    dominant_freq = float(freqs[np.argmax(psd)])
    low_freq_share = float(psd[freqs <= 1000].sum() / psd_sum)

    return centroid, bandwidth, rolloff_95, dominant_freq, low_freq_share


def basic_signal_features(path):
    sample_rate, x = load_first_channel(path)
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    zcr = float(np.mean(x[:-1] * x[1:] < 0))
    centroid, bandwidth, rolloff_95, dominant_freq, low_freq_share = spectral_features(x, sample_rate)
    return {
        "sample_rate": sample_rate,
        "rms": rms,
        "peak_abs": peak,
        "crest_factor": peak / max(rms, 1e-8),
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "rolloff_95_hz": rolloff_95,
        "dominant_freq_hz": dominant_freq,
        "low_freq_share_le_1khz": low_freq_share,
    }


def savefig(name):
    path = REF_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


metadata_df = scan_dataset(MACHINE_TYPES)
metadata_df["duration_minutes"] = metadata_df["duration_seconds"] / 60

summary = (
    metadata_df.groupby(["machine_type", "machine_id", "status"], as_index=False)
    .agg(clips=("path", "count"), total_minutes=("duration_minutes", "sum"))
)

machine_summary = (
    metadata_df.groupby(["machine_type", "status"], as_index=False)
    .agg(clips=("path", "count"), total_minutes=("duration_minutes", "sum"))
)

imbalance = (
    summary.pivot(index=["machine_type", "machine_id"], columns="status", values="clips")
    .fillna(0)
    .reset_index()
)
imbalance["abnormal_share"] = imbalance["abnormal"] / (imbalance["normal"] + imbalance["abnormal"])
imbalance["normal_to_abnormal_ratio"] = imbalance["normal"] / imbalance["abnormal"]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.barplot(
    data=machine_summary,
    x="machine_type",
    y="clips",
    hue="status",
    palette=PALETTE,
    ax=axes[0],
)
axes[0].set_title("Clip Counts by Machine Type")
axes[0].set_xlabel("Machine type")
axes[0].set_ylabel("Number of 10-second clips")

sns.barplot(
    data=machine_summary,
    x="machine_type",
    y="total_minutes",
    hue="status",
    palette=PALETTE,
    ax=axes[1],
)
axes[1].set_title("Recorded Minutes by Machine Type")
axes[1].set_xlabel("Machine type")
axes[1].set_ylabel("Minutes of audio")

for ax in axes:
    ax.legend(title="Condition")

savefig("machine_counts_and_minutes.png")

heatmap_df = summary.pivot_table(
    index=["machine_type", "machine_id"], columns="status", values="clips", fill_value=0
)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Clip Counts by Machine Model and Condition")
plt.xlabel("Condition")
plt.ylabel("Machine type / model ID")
savefig("machine_model_condition_heatmap.png")

plt.figure(figsize=(12, 6))
imbalance_sorted = imbalance.sort_values(
    ["abnormal_share", "machine_type", "machine_id"], ascending=[False, True, True]
).copy()
imbalance_sorted["group"] = imbalance_sorted["machine_type"] + "/" + imbalance_sorted["machine_id"]
sns.barplot(data=imbalance_sorted, x="group", y="abnormal_share", color="#F58518")
plt.axhline(
    metadata_df.status.eq("abnormal").mean(),
    color="black",
    linestyle="--",
    label="Overall abnormal share",
)
plt.title("Abnormal Share by Machine Model")
plt.xlabel("Machine type / model ID")
plt.ylabel("Abnormal share")
plt.xticks(rotation=45)
plt.legend()
savefig("abnormal_share_by_group.png")

random_state = np.random.RandomState(RANDOM_SEED)
sampled_df = (
    metadata_df.assign(_rand=random_state.rand(len(metadata_df)))
    .sort_values("_rand")
    .groupby(["machine_type", "machine_id", "status"], group_keys=False)
    .head(15)
    .drop(columns="_rand")
    .reset_index(drop=True)
)

feature_rows = []
for row in sampled_df.itertuples(index=False):
    features = basic_signal_features(Path(row.path))
    features.update(
        {
            "path": row.path,
            "machine_type": row.machine_type,
            "machine_id": row.machine_id,
            "status": row.status,
        }
    )
    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
sns.boxplot(data=features_df, x="machine_type", y="rms", hue="status", palette=PALETTE, ax=axes[0, 0])
axes[0, 0].set_title("Signal Energy by Machine Type")
axes[0, 0].set_xlabel("Machine type")
axes[0, 0].set_ylabel("RMS energy")

sns.boxplot(
    data=features_df,
    x="machine_type",
    y="zero_crossing_rate",
    hue="status",
    palette=PALETTE,
    ax=axes[0, 1],
)
axes[0, 1].set_title("Waveform Roughness / Noisiness")
axes[0, 1].set_xlabel("Machine type")
axes[0, 1].set_ylabel("Zero-crossing rate")

sns.boxplot(
    data=features_df,
    x="machine_type",
    y="spectral_centroid_hz",
    hue="status",
    palette=PALETTE,
    ax=axes[1, 0],
)
axes[1, 0].set_title("Spectral Centroid by Machine Type")
axes[1, 0].set_xlabel("Machine type")
axes[1, 0].set_ylabel("Spectral centroid (Hz)")

sns.boxplot(
    data=features_df,
    x="machine_type",
    y="low_freq_share_le_1khz",
    hue="status",
    palette=PALETTE,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Power Share in Low Frequencies")
axes[1, 1].set_xlabel("Machine type")
axes[1, 1].set_ylabel("Power share <= 1 kHz")

for ax in axes.flat:
    ax.legend(title="Condition")

savefig("signal_feature_boxplots.png")

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=features_df,
    x="spectral_centroid_hz",
    y="rms",
    hue="status",
    style="machine_type",
    palette=PALETTE,
    s=90,
)
plt.title("Sampled Clips in a Simple Feature Space")
plt.xlabel("Spectral centroid (Hz)")
plt.ylabel("RMS energy")
savefig("sample_feature_scatter.png")

example_rows = []
for machine_type in MACHINE_TYPES:
    for status in ["normal", "abnormal"]:
        subset = metadata_df[
            (metadata_df.machine_type == machine_type) & (metadata_df.status == status)
        ]
        if len(subset) == 0:
            continue
        example_rows.append(subset.sample(1, random_state=RANDOM_SEED).iloc[0])

example_df = pd.DataFrame(example_rows).reset_index(drop=True)

fig, axes = plt.subplots(len(example_df), 2, figsize=(16, 4 * len(example_df)))
if len(example_df) == 1:
    axes = np.array([axes])

for i, row in enumerate(example_df.itertuples(index=False)):
    sample_rate, x = load_first_channel(Path(row.path))
    t = np.arange(len(x)) / sample_rate
    freqs, times, sxx = signal.spectrogram(x, fs=sample_rate, nperseg=1024, noverlap=512)
    sxx_db = 10 * np.log10(sxx + 1e-10)

    axes[i, 0].plot(t, x, color=PALETTE[row.status], linewidth=0.8)
    axes[i, 0].set_title(f"Waveform: {row.machine_type} | {row.machine_id} | {row.status}")
    axes[i, 0].set_xlabel("Time (s)")
    axes[i, 0].set_ylabel("Normalized amplitude")

    im = axes[i, 1].pcolormesh(times, freqs, sxx_db, shading="gouraud", cmap="magma")
    axes[i, 1].set_title(f"Spectrogram: {row.machine_type} | {row.machine_id} | {row.status}")
    axes[i, 1].set_xlabel("Time (s)")
    axes[i, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

savefig("waveform_and_spectrogram_examples.png")

