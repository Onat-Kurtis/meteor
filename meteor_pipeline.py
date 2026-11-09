import os
import sys
import time
import wave
import socket
import struct
import datetime
import argparse
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt

# GRAVES Radar Frekansı: 143.050 MHz (USB modunda ~1000 Hz ses tonu hedeflenir)
SAMPLE_RATE = 12000
RADAR_FREQ_HZ = 143.05e6
LAMBDA_M = 3.0e8 / RADAR_FREQ_HZ  # ~2.097 metre

os.makedirs("data/events_wav", exist_ok=True)
os.makedirs("data/spectrograms", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)
CSV_PATH = "data/meteor_catalog.csv"

def bandpass_filter(data, lowcut=600.0, highcut=1600.0, fs=SAMPLE_RATE, order=5):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.lfilter(b, a, data)

def extract_physics_and_save(audio_clip, fs, shower_label, event_id, timestamp_str):
    # 1. Ham WAV Kaydı
    wav_filename = f"data/events_wav/{shower_label}_{event_id}.wav"
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes((audio_clip * 32767).astype(np.int16).tobytes())

    # 2. STFT (Zaman-Frekans Matrisi)
    f, t, Zxx = signal.stft(audio_clip, fs=fs, nperseg=512, noverlap=384)
    freq_mask = (f >= 500) & (f <= 1800)
    f_roi = f[freq_mask]
    power_spec = np.abs(Zxx[freq_mask, :]) ** 2
    power_db = 10 * np.log10(power_spec + 1e-12)

    # 3. Gürültü Tabanı ve Tepe SNR
    noise_floor_db = np.percentile(power_db, 20)
    peak_power_db = np.max(power_db)
    peak_snr_db = peak_power_db - noise_floor_db
    snr_linear = 10 ** (peak_snr_db / 10.0)

    # 4. Sönümlenme Süresi (Tau) ve Plazma Rejimi
    time_profile = np.max(power_db, axis=0)
    peak_idx = np.argmax(time_profile)
    above_thresh = np.where(time_profile > (noise_floor_db + 6.0))[0]
    duration_sec = (t[above_thresh[-1]] - t[above_thresh[0]]) if len(above_thresh) > 1 else 0.08

    # Tepe noktasından 1/e (~4.34 dB) düşüş süresi
    post_peak = time_profile[peak_idx:]
    decay_indices = np.where(post_peak <= (peak_power_db - 4.34))[0]
    decay_tau_sec = (t[peak_idx + decay_indices[0]] - t[peak_idx]) if len(decay_indices) > 0 else duration_sec * 0.5
    decay_tau_sec = max(decay_tau_sec, 0.02)

    plasma_regime = "Overdense" if decay_tau_sec >= 0.45 else "Underdense"
    estimated_q = 2.4e14 * (decay_tau_sec / 0.45) ** 1.5

    # 5. Ambipolar Difüzyon Katsayısı (D_a) ve Tahmini İrtifa (h)
    diffusion_coeff = (LAMBDA_M ** 2) / (16.0 * (np.pi ** 2) * decay_tau_sec)
    # Standart üst atmosfer ampirik D_a - irtifa bağıntısı (80 - 115 km)
    est_altitude_km = np.clip(93.5 + 6.2 * np.log10(diffusion_coeff / 4.0), 75.0, 120.0)

    # 6. Başlangıç Doppler Kayması (Hız) ve Spektral Genişlik (Parçalanma)
    onset_slice = power_spec[:, max(0, peak_idx - 3):min(power_spec.shape[1], peak_idx + 2)]
    dominant_freqs = f_roi[np.argmax(onset_slice, axis=0)]
    doppler_shift_hz = np.ptp(dominant_freqs) + abs( dominant_freqs[0] - 1000.0 )
    radial_vel_kms = (3.0e5 * doppler_shift_hz) / (2.0 * RADAR_FREQ_HZ) * 1e3 # Efektif ölçekli hız

    peak_freq_spectrum = power_spec[:, peak_idx]
    half_max = np.max(peak_freq_spectrum) * 0.5
    spectral_width_hz = np.sum(peak_freq_spectrum > half_max) * (f_roi[1] - f_roi[0])

    # 7. Haberleşme Verimlilik Skoru (MCI)
    comm_score = duration_sec * np.log2(1.0 + snr_linear)

    # 8. Spektrogram PNG Kaydı
    plt.figure(figsize=(6, 4), dpi=120)
    plt.pcolormesh(t, f_roi, power_db, shading='gouraud', cmap='inferno', vmin=noise_floor_db, vmax=peak_power_db)
    plt.title(f"{shower_label} | ID: {event_id} | SNR: {peak_snr_db:.1f} dB | Tau: {decay_tau_sec:.2f}s")
    plt.ylabel("Frekans (Hz)")
    plt.xlabel("Zaman (s)")
    plt.colorbar(label="Güç (dB)")
    plt.tight_layout()
    plt.savefig(f"data/spectrograms/{shower_label}_{event_id}.png")
    plt.close()

    # 9. CSV'ye Satır Ekleme
    row = {
        "event_id": f"{shower_label}_{event_id}",
        "shower": shower_label,
        "timestamp_utc": timestamp_str,
        "noise_floor_db": round(noise_floor_db, 2),
        "peak_snr_db": round(peak_snr_db, 2),
        "duration_sec": round(duration_sec, 3),
        "decay_tau_sec": round(decay_tau_sec, 3),
        "doppler_shift_hz": round(doppler_shift_hz, 2),
        "radial_vel_kms": round(radial_vel_kms, 2),
        "spectral_width_hz": round(spectral_width_hz, 2),
        "diffusion_coeff_m2s": round(diffusion_coeff, 4),
        "est_altitude_km": round(est_altitude_km, 2),
        "plasma_regime": plasma_regime,
        "estimated_q": f"{estimated_q:.2e}",
        "comm_score": round(comm_score, 2)
    }
    df_row = pd.DataFrame([row])
    df_row.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)

def run_capture(shower_label, duration_hours, stream_host, stream_port, start_utc=0, end_utc=24):
    end_time = time.time() + int(duration_hours * 3600)
    event_counter = int(time.time()) % 100000
    chunk_samples = SAMPLE_RATE * 5  # 5 saniyelik pencereler

    while time.time() < end_time:
        current_utc_hour = datetime.datetime.utcnow().hour

        if start_utc <= current_utc_hour < end_utc:
            pass
        elif (start_utc - current_utc_hour) % 24 <= 5:
            time.sleep(20)
            continue
        else:
            break

        try:
            raw_audio = np.random.normal(0, 0.02, chunk_samples)
            if stream_host:
                with socket.create_connection((stream_host, stream_port), timeout=10) as sock:
                    raw_bytes = sock.recv(chunk_samples * 2)
                    if len(raw_bytes) == chunk_samples * 2:
                        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            time.sleep(2)
            continue

        filtered = bandpass_filter(raw_audio)
        envelope = np.abs(signal.hilbert(filtered))
        median_env = np.median(envelope)
        std_env = np.std(envelope)

        if np.max(envelope) > (median_env + 4.2 * std_env):
            event_counter += 1
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            extract_physics_and_save(filtered, SAMPLE_RATE, shower_label, event_counter, ts)
        time.sleep(0.1)

def run_final_analysis():
    if not os.path.exists(CSV_PATH):
        return
    df = pd.read_csv(CSV_PATH)
    taurids = df[df["shower"] == "Taurid"]
    leonids = df[df["shower"] == "Leonid"]

    if len(taurids) > 2 and len(leonids) > 2:
        ks_tau = stats.ks_2samp(taurids["decay_tau_sec"], leonids["decay_tau_sec"])
        ks_vel = stats.ks_2samp(taurids["doppler_shift_hz"], leonids["doppler_shift_hz"])

        # 1. ECDF ve K-S Test Grafiği
        plt.figure(figsize=(10, 4), dpi=150)
        plt.subplot(1, 2, 1)
        plt.ecdf(taurids["decay_tau_sec"], label="Taurid (Yüksek Kütle)")
        plt.ecdf(leonids["decay_tau_sec"], label="Leonid (Düşük Kütle)")
        plt.title(f"Sönümlenme Süresi ECDF\nK-S D={ks_tau.statistic:.3f}, p={ks_tau.pvalue:.2e}")
        plt.xlabel("Tau (s)")
        plt.legend()

        # 2. 2x2 Faz Uzayı (Hız vs Kütle/Tau)
        plt.subplot(1, 2, 2)
        for label, grp in df.groupby("shower"):
            plt.scatter(grp["doppler_shift_hz"], grp["decay_tau_sec"], label=label, alpha=0.7)
        plt.yscale("log")
        plt.xlabel("Başlangıç Doppler Kayması Δf (Hz) [Hız]")
        plt.ylabel("Sönümlenme Süresi τ (s) [Kütle]")
        plt.title("2x2 Kinetik Faz Uzayı")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/plots/KS_and_PhaseSpace_Summary.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shower", type=str, default="Taurid")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--host", type=str, default="")
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--start-utc", type=int, default=0, help="Kayıt başlangıç UTC saati")
    parser.add_argument("--end-utc", type=int, default=24, help="Kayıt bitiş UTC saati")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        run_final_analysis()
    else:
        run_capture(args.shower, args.hours, args.host, args.port, args.start_utc, args.end_utc)
