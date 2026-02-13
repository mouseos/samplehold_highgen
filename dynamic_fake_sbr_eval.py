import argparse
import os
import sys
import time
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import medfilt, welch, butter, sosfilt
from scipy.interpolate import interp1d

# ★高速化のためのJITコンパイラ
from numba import jit, float64

# --- グローバル設定 ---
ORIGINAL_SR = 48000

# --- 解析パラメータ ---
N_FFT = 512
HOP_LENGTH = 512

# カットオフ検出パラメータ
PASSBAND_START_HZ = 500         
PASSBAND_END_HZ = 1500          
DROP_THRESHOLD_DB = 11.0        
ABSOLUTE_SILENCE_DB = -70.0    

# ★スライド「手法③」固有パラメータ
ENERGY_PERCENTILE = 0.9999   # 累積エネルギー99.99%地点
SCAN_RANGE_HZ = 4000         # 前後探索範囲

# ★ジャズ対策用追加パラメータ
DETECTION_LOW_CUT_HZ = 1000  

# --- 平滑化パラメータ ---
SPIKE_REJECTION_FILTER_SIZE = 31
HYSTERESIS_THRESHOLD_HZ = 1200
SUSTAIN_FRAMES = 60
ATTACK_COEFF = 0.15
RELEASE_COEFF = 0.005
SILENCE_THRESHOLD_HZ = 1000

# --- 自動ゲイン算出用パラメータ ---
GAIN_SMOOTH_WINDOW = 120
GAIN_CALC_LOW_CUT_HZ = 2000
MAX_GAIN_LIMIT_DB = 0.0        

# 感度設定 (Dual Shelf)
SENSITIVITY_MID = 0.2
SENSITIVITY_HIGH = 0.25

# ==========================================
# ★ Numbaによる高速化されたサンプル単位処理
# ==========================================

@jit(nopython=True)
def apply_dynamic_processing(audio, cutoffs_per_sample, tilts_db_per_sample, sr):
    n_samples = len(audio)
    output = np.zeros_like(audio)
    sh_output = np.zeros_like(audio)
    
    hold_val = 0.0
    phase_accumulator = 0.0
    
    z1_m = 0.0; z2_m = 0.0
    z1_h = 0.0; z2_h = 0.0
    
    pi = 3.141592653589793
    
    for i in range(n_samples):
        fc = cutoffs_per_sample[i]
        tilt_db = tilts_db_per_sample[i]
        
        if fc < 100.0: fc = 100.0
        if fc > sr / 2.0 - 100.0: fc = sr / 2.0 - 100.0
        
        # A. Variable Sample & Hold
        target_sr = fc * 2.0
        phase_step = target_sr / sr
        phase_accumulator += phase_step
        
        if phase_accumulator >= 1.0:
            hold_val = audio[i]
            phase_accumulator -= 1.0
        
        exciter_sig = hold_val
        sh_output[i] = exciter_sig
        
        if tilt_db < -60.0:
            output[i] = audio[i]
            z1_m *= 0.99; z2_m *= 0.99
            z1_h *= 0.99; z2_h *= 0.99
            continue

        # B. Dual Dynamic High Shelf Filter
        # 1. Mid Shelf
        mid_gain = tilt_db * SENSITIVITY_MID
        A_m = 10.0 ** (mid_gain / 40.0)
        w0_m = 2.0 * pi * fc / sr
        cos_w0_m = np.cos(w0_m)
        sin_w0_m = np.sin(w0_m)
        alpha_m = sin_w0_m / (2.0 * 0.707)
        sqA_m = np.sqrt(A_m)
        
        a0_m = (A_m+1) - (A_m-1)*cos_w0_m + 2*sqA_m*alpha_m
        b0_m = A_m * ((A_m+1) + (A_m-1)*cos_w0_m + 2*sqA_m*alpha_m)
        b1_m = -2 * A_m * ((A_m-1) + (A_m+1)*cos_w0_m)
        b2_m = A_m * ((A_m+1) + (A_m-1)*cos_w0_m - 2*sqA_m*alpha_m)
        a1_m = 2 * ((A_m-1) - (A_m+1)*cos_w0_m)
        a2_m = (A_m+1) - (A_m-1)*cos_w0_m - 2*sqA_m*alpha_m
        b0_m /= a0_m; b1_m /= a0_m; b2_m /= a0_m; a1_m /= a0_m; a2_m /= a0_m
        
        mid_out = b0_m * exciter_sig + z1_m
        z1_m = b1_m * exciter_sig - a1_m * mid_out + z2_m
        z2_m = b2_m * exciter_sig - a2_m * mid_out
        
        # 2. High Shelf
        high_gain = tilt_db * SENSITIVITY_HIGH
        high_fc = fc * 2.0
        if high_fc > sr / 2.0 - 100.0: high_fc = sr / 2.0 - 100.0
        
        A_h = 10.0 ** (high_gain / 40.0)
        w0_h = 2.0 * pi * high_fc / sr
        cos_w0_h = np.cos(w0_h)
        sin_w0_h = np.sin(w0_h)
        alpha_h = sin_w0_h / (2.0 * 0.707)
        sqA_h = np.sqrt(A_h)
        
        a0_h = (A_h+1) - (A_h-1)*cos_w0_h + 2*sqA_h*alpha_h
        b0_h = A_h * ((A_h+1) + (A_h-1)*cos_w0_h + 2*sqA_h*alpha_h)
        b1_h = -2 * A_h * ((A_h-1) + (A_h+1)*cos_w0_h)
        b2_h = A_h * ((A_h+1) + (A_h-1)*cos_w0_h - 2*sqA_h*alpha_h)
        a1_h = 2 * ((A_h-1) - (A_h+1)*cos_w0_h)
        a2_h = (A_h+1) - (A_h-1)*cos_w0_h - 2*sqA_h*alpha_h
        b0_h /= a0_h; b1_h /= a0_h; b2_h /= a0_h; a1_h /= a0_h; a2_h /= a0_h
        
        final_out = b0_h * mid_out + z1_h
        z1_h = b1_h * mid_out - a1_h * final_out + z2_h
        z2_h = b2_h * mid_out - a2_h * final_out
        
        output[i] = final_out

    return output, sh_output

# ==========================================
# 評価用劣化フィルタ
# ==========================================
def apply_lowpass_filter(y, sr, cutoff_freq, order=8):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff_freq / nyquist
    sos = butter(order, norm_cutoff, btype='low', analog=False, output='sos')
    if y.ndim == 1:
        y_filtered = sosfilt(sos, y)
    else:
        y_filtered = np.zeros_like(y)
        for ch in range(y.shape[1]):
            y_filtered[:, ch] = sosfilt(sos, y[:, ch])
    return y_filtered

# ==========================================
# 解析ロジック
# ==========================================
def apply_moving_average(data, window_size):
    if len(data) < window_size: return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def apply_hysteresis_smoothing(cutoffs_despiked, attack_coeff, release_coeff):
    if len(cutoffs_despiked) == 0: return cutoffs_despiked
    smoothed_cutoffs = np.zeros_like(cutoffs_despiked)
    valid_values = cutoffs_despiked[cutoffs_despiked > SILENCE_THRESHOLD_HZ]
    if len(valid_values) > 0:
        representative_val = np.percentile(valid_values, 10)
    else:
        representative_val = 5000.0
    smoothed_cutoffs[0] = representative_val
    trend = 0
    trend_sustain_count = 0
    CLAMP_FRAMES = 5 
    for i in range(1, len(cutoffs_despiked)):
        if i < CLAMP_FRAMES:
            smoothed_cutoffs[i] = representative_val
            continue
        current_raw = cutoffs_despiked[i]
        previous_smoothed = smoothed_cutoffs[i-1]
        diff = current_raw - previous_smoothed
        current_trend = 0
        if diff > HYSTERESIS_THRESHOLD_HZ: current_trend = 1
        elif diff < -HYSTERESIS_THRESHOLD_HZ: current_trend = -1
        if current_trend == trend and current_trend != 0: trend_sustain_count += 1
        else:
            trend = current_trend
            trend_sustain_count = 1
        is_recovering_from_silence = previous_smoothed < SILENCE_THRESHOLD_HZ and trend == 1
        if trend == 0 or (trend_sustain_count < SUSTAIN_FRAMES and not is_recovering_from_silence):
            smoothed_cutoffs[i] = previous_smoothed
            continue
        coeff = attack_coeff if trend == 1 else release_coeff
        smoothed_cutoffs[i] = previous_smoothed * (1 - coeff) + current_raw * coeff
    return smoothed_cutoffs

def detect_cutoff_and_slope(y, sr):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_abs = np.abs(S)
    S_energy = S_abs ** 2  
    S_db = librosa.amplitude_to_db(S_abs, ref=np.max)
    
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_resolution = sr / N_FFT
    
    cutoff_frequencies = []
    tilt_values = [] 

    passband_start_idx = np.searchsorted(fft_freqs, PASSBAND_START_HZ)
    passband_end_idx = np.searchsorted(fft_freqs, PASSBAND_END_HZ)
    low_cut_bin_idx_for_tilt = np.searchsorted(fft_freqs, GAIN_CALC_LOW_CUT_HZ)
    search_range_bins = int(SCAN_RANGE_HZ / freq_resolution)

    # ジャズ対策
    det_low_cut_idx = np.searchsorted(fft_freqs, DETECTION_LOW_CUT_HZ)

    for frame_idx in range(S_db.shape[1]):
        frame_db = S_db[:, frame_idx]
        frame_energy = S_energy[:, frame_idx]
        
        if np.max(frame_db) < ABSOLUTE_SILENCE_DB:
            cutoff_frequencies.append(np.nan)
            tilt_values.append(np.nan)
            continue
        
        # 累積エネルギー計算（低域マスク）
        calc_energy = frame_energy.copy()
        calc_energy[:det_low_cut_idx] = 0.0 
        
        total_energy_filtered = np.sum(calc_energy)
        
        if total_energy_filtered < 1e-9:
            cutoff_frequencies.append(np.nan); tilt_values.append(np.nan); continue

        cumulative_energy = np.cumsum(calc_energy)
        threshold_energy = total_energy_filtered * ENERGY_PERCENTILE
        provisional_idx = np.searchsorted(cumulative_energy, threshold_energy)
        
        ref_level_db = np.mean(frame_db[passband_start_idx:passband_end_idx])
        drop_target_db = ref_level_db - DROP_THRESHOLD_DB
        
        detected_cutoff_freq = np.nan
        search_start_idx = min(len(frame_db) - 1, provisional_idx + search_range_bins)
        search_end_idx = max(passband_end_idx, provisional_idx - search_range_bins)
        
        for i in range(search_start_idx, search_end_idx, -1):
            if frame_db[i] > drop_target_db:
                detected_cutoff_freq = fft_freqs[i]
                break
        
        cutoff_frequencies.append(detected_cutoff_freq)

        # Tilt計算
        if not np.isnan(detected_cutoff_freq):
            current_cutoff_bin = np.searchsorted(fft_freqs, detected_cutoff_freq)
            if current_cutoff_bin > low_cut_bin_idx_for_tilt + 2:
                valid_freqs = fft_freqs[low_cut_bin_idx_for_tilt:current_cutoff_bin]
                valid_dbs = frame_db[low_cut_bin_idx_for_tilt:current_cutoff_bin]
                mask = valid_freqs > 0
                if np.sum(mask) > 1:
                    x = np.log10(valid_freqs[mask])
                    y_vals = valid_dbs[mask]
                    slope, intercept = np.polyfit(x, y_vals, 1)
                    val_low = slope * np.log10(GAIN_CALC_LOW_CUT_HZ)
                    val_high = slope * np.log10(detected_cutoff_freq)
                    tilt_db = val_high - val_low
                    tilt_values.append(tilt_db)
                else: tilt_values.append(np.nan)
            else: tilt_values.append(np.nan)
        else: tilt_values.append(np.nan)

    times = librosa.frames_to_time(np.arange(len(cutoff_frequencies)), sr=sr, hop_length=HOP_LENGTH)
    def fill_nan(arr):
        mask = np.isnan(arr)
        if not np.any(mask): return arr
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        out = arr[idx]
        if np.isnan(out[0]):
            if np.all(np.isnan(out)): return np.full_like(out, 5000.0) 
            first_valid_idx = np.where(~np.isnan(out))[0][0]
            first_valid_val = out[first_valid_idx]
            out[:first_valid_idx] = first_valid_val
        return out
    
    cutoffs_np = fill_nan(np.array(cutoff_frequencies))
    tilts_np = fill_nan(np.array(tilt_values))
    
    if len(cutoffs_np) > SPIKE_REJECTION_FILTER_SIZE:
        cutoffs_despiked = medfilt(cutoffs_np, kernel_size=SPIKE_REJECTION_FILTER_SIZE)
    else: cutoffs_despiked = cutoffs_np.copy()
    
    cutoffs_smoothed = apply_hysteresis_smoothing(cutoffs_despiked, ATTACK_COEFF, RELEASE_COEFF)
    
    if len(tilts_np) > SPIKE_REJECTION_FILTER_SIZE:
        tilts_despiked = medfilt(tilts_np, kernel_size=SPIKE_REJECTION_FILTER_SIZE)
    else: tilts_despiked = tilts_np.copy()
    
    tilts_smoothed = apply_moving_average(tilts_despiked, GAIN_SMOOTH_WINDOW)
    tilts_smoothed = np.clip(tilts_smoothed, -60.0, MAX_GAIN_LIMIT_DB)
    
    return times, cutoffs_smoothed, tilts_smoothed

# ==========================================
# プロット関数 (メモリ対策済み)
# ==========================================

def save_individual_plot(fig, base_filename, suffix):
    """個別プロットを保存するためのヘルパー関数"""
    out_img_name = base_filename.replace('.wav', suffix + '.png')
    fig.savefig(out_img_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {out_img_name}")
    plt.close(fig)

def plot_paper_evaluation(original_y, degraded_y, sh_y, enhanced_y, sr, times, cutoffs, tilts, eval_cutoff, out_filename):
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    # --- 共通データ準備 ---
    deg_plot = degraded_y[:, 0] if degraded_y.ndim > 1 else degraded_y
    enh_plot = enhanced_y[:, 0] if enhanced_y.ndim > 1 else enhanced_y
    
    # スペクトログラム計算
    S_deg = librosa.amplitude_to_db(np.abs(librosa.stft(deg_plot, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
    S_enh = librosa.amplitude_to_db(np.abs(librosa.stft(enh_plot, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
    
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    center_frame_idx_raw = S_deg.shape[1] // 2 # 生データ時点での中心

    # =========================================
    # ★メモリ対策: 長時間ファイル用の間引き処理
    # =========================================
    MAX_PLOT_WIDTH = 15000  # 描画する最大フレーム数
    
    total_frames = S_deg.shape[1]
    step = 1
    
    # 描画用変数の初期化
    plot_S_deg = S_deg
    plot_S_enh = S_enh
    plot_times = times
    plot_cutoffs = cutoffs
    plot_tilts = tilts
    plot_hop_length = HOP_LENGTH
    
    if total_frames > MAX_PLOT_WIDTH:
        step = int(np.ceil(total_frames / MAX_PLOT_WIDTH))
        print(f"Warning: Audio is too long ({total_frames} frames). Downsampling plots by 1/{step} to save memory...")
        
        # 配列を間引く (スライス [::step])
        plot_S_deg = S_deg[:, ::step]
        plot_S_enh = S_enh[:, ::step]
        
        # オーバーレイ用の時間軸データも合わせる
        min_len = min(plot_S_deg.shape[1], len(times[::step]), len(cutoffs[::step]), len(tilts[::step]))
        
        plot_times = times[::step][:min_len]
        plot_cutoffs = cutoffs[::step][:min_len]
        plot_tilts = tilts[::step][:min_len]
        plot_S_deg = plot_S_deg[:, :min_len]
        plot_S_enh = plot_S_enh[:, :min_len]
        
        # Specshow用のHop Lengthを補正
        plot_hop_length = HOP_LENGTH * step

    # =========================================
    # Part 1: 個別プロット
    # =========================================
    FIG_SIZE_SINGLE = (8, 6)
    FIG_SIZE_DOUBLE = (16, 6)

    # (A) Input Spectrogram
    fig_a = plt.figure(figsize=FIG_SIZE_SINGLE)
    ax_a = fig_a.add_subplot(111)
    img_a = librosa.display.specshow(plot_S_deg, sr=sr, hop_length=plot_hop_length, x_axis='time', y_axis='hz', ax=ax_a, cmap='magma')
    ax_a.plot(plot_times, plot_cutoffs, color='cyan', linewidth=2.0, linestyle='--', label='Detected Cutoff')
    ax_a.set_title(f'(a) Input Spectrogram (LP: {eval_cutoff}Hz)', fontweight='bold')
    ax_a.set_xlabel('Time (s)')
    ax_a.set_ylabel('Frequency (Hz)')
    ax_a.legend(loc='upper right')
    fig_a.colorbar(img_a, ax=ax_a, format='%+2.0f dB', label='Magnitude (dB)')
    save_individual_plot(fig_a, out_filename, "_plot_A_input_spec")

    # (B) Output Spectrogram
    fig_b = plt.figure(figsize=FIG_SIZE_SINGLE)
    ax_b = fig_b.add_subplot(111)
    img_b = librosa.display.specshow(plot_S_enh, sr=sr, hop_length=plot_hop_length, x_axis='time', y_axis='hz', ax=ax_b, cmap='magma')
    ax_b.set_title('(b) Restored Spectrogram', fontweight='bold')
    ax_b.set_xlabel('Time (s)')
    ax_b.set_ylabel('Frequency (Hz)')
    fig_b.colorbar(img_b, ax=ax_b, format='%+2.0f dB', label='Magnitude (dB)')
    save_individual_plot(fig_b, out_filename, "_plot_B_output_spec")

    # (C) Tilt Evolution
    fig_c = plt.figure(figsize=FIG_SIZE_SINGLE)
    ax_c = fig_c.add_subplot(111)
    ax_c.plot(plot_times, plot_tilts, color='#6a0dad', linewidth=2.0)
    ax_c.set_title('(c) Spectral Tilt Evolution', fontweight='bold')
    ax_c.set_ylabel('Tilt (dB)')
    ax_c.set_xlabel('Time (s)')
    ax_c.set_ylim(-50, 5)
    ax_c.grid(True, alpha=0.5, linestyle='--')
    save_individual_plot(fig_c, out_filename, "_plot_C_tilt_evo")

    # (D) Linear Regression @ Center Frame
    fig_d = plt.figure(figsize=FIG_SIZE_SINGLE)
    ax_d = fig_d.add_subplot(111)
    
    # 生データの中心フレームを取得
    frame_db = S_deg[:, center_frame_idx_raw]
    center_cutoff = cutoffs[min(center_frame_idx_raw, len(cutoffs)-1)]
    low_cut_idx = np.searchsorted(fft_freqs, GAIN_CALC_LOW_CUT_HZ)
    cutoff_idx = np.searchsorted(fft_freqs, center_cutoff)
    
    ax_d.semilogx(fft_freqs, frame_db, color='gray', alpha=0.6, label='Raw Spectrum')
    
    calc_tilt = 0.0
    if cutoff_idx > low_cut_idx + 2:
        valid_freqs = fft_freqs[low_cut_idx:cutoff_idx]
        valid_dbs = frame_db[low_cut_idx:cutoff_idx]
        mask = valid_freqs > 0
        if np.sum(mask) > 1:
            x_log = np.log10(valid_freqs[mask])
            y_vals = valid_dbs[mask]
            slope, intercept = np.polyfit(x_log, y_vals, 1)
            
            x_plot = np.logspace(np.log10(GAIN_CALC_LOW_CUT_HZ), np.log10(center_cutoff), 200)
            y_plot = slope * np.log10(x_plot) + intercept
            
            ax_d.semilogx(x_plot, y_plot, color='red', linewidth=2.5, label='Regression Fit')
            
            x_ext = np.logspace(np.log10(center_cutoff), np.log10(sr/2), 100)
            y_ext = slope * np.log10(x_ext) + intercept
            ax_d.semilogx(x_ext, y_ext, color='red', linewidth=1.5, linestyle=':', alpha=0.8)
            
            calc_tilt = (slope * np.log10(center_cutoff)) - (slope * np.log10(GAIN_CALC_LOW_CUT_HZ))
            ax_d.text(0.05, 0.1, f'Calculated Tilt: {calc_tilt:.2f} dB', transform=ax_d.transAxes, 
                      bbox=dict(facecolor='white', alpha=0.9))

    ax_d.axvline(x=GAIN_CALC_LOW_CUT_HZ, color='green', linestyle=':', label='Fit Start (2kHz)')
    ax_d.axvline(x=center_cutoff, color='blue', linestyle=':', label='Cutoff')
    
    ax_d.set_xscale('log')
    ax_d.set_xticks([100, 500, 1000, 5000, 10000, 20000])
    ax_d.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_d.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    
    ax_d.set_title(f'(d) Linear Regression @ Center Frame ({times[center_frame_idx_raw]:.2f}s)', fontweight='bold')
    ax_d.set_xlabel('Frequency (Hz)')
    ax_d.set_ylabel('Magnitude (dB)')
    ax_d.set_xlim(100, sr/2)
    ax_d.set_ylim(-80, 5)
    ax_d.grid(True, which="both", ls="-", alpha=0.3)
    ax_d.legend(loc='upper right', fontsize=10)
    
    save_individual_plot(fig_d, out_filename, "_plot_D_regression")

    # (E) Overall PSD
    fig_e = plt.figure(figsize=FIG_SIZE_DOUBLE)
    ax_e = fig_e.add_subplot(111)
    y_orig = original_y[:, 0] if original_y.ndim > 1 else original_y
    y_sh = sh_y[:, 0] if sh_y.ndim > 1 else sh_y
    N_PER_SEG = 8192 
    f, P_orig = welch(y_orig, sr, nperseg=N_PER_SEG)
    _, P_deg = welch(deg_plot, sr, nperseg=N_PER_SEG)
    _, P_sh = welch(y_sh, sr, nperseg=N_PER_SEG)
    _, P_enh = welch(enh_plot, sr, nperseg=N_PER_SEG)
    
    ax_e.plot(f, 10*np.log10(P_orig+1e-12), color='green', alpha=0.5, label='Original', linewidth=1.5)
    ax_e.plot(f, 10*np.log10(P_deg+1e-12), color='blue', alpha=0.7, label='Degraded/Input', linewidth=1.5)
    ax_e.plot(f, 10*np.log10(P_sh+1e-12), color='orange', linestyle='--', label='S&H Raw', linewidth=1.5, alpha=0.8)
    ax_e.plot(f, 10*np.log10(P_enh+1e-12), color='red', label='Restored', linewidth=2.0)
    
    ax_e.set_title('(e) Overall Power Spectral Density (PSD)', fontweight='bold')
    ax_e.set_xlabel('Frequency (Hz)')
    ax_e.set_ylabel('PSD (dB/Hz)')
    ax_e.set_xlim(0, 24000)
    ax_e.set_ylim(bottom=-110)
    ax_e.legend(loc='lower left', ncol=4, fontsize=10)
    ax_e.grid(True, which='both', alpha=0.3)
    save_individual_plot(fig_e, out_filename, "_plot_E_psd")


    # =========================================
    # Part 2: 全体まとめプロット (16:9)
    # =========================================
    fig_comb = plt.figure(figsize=(16, 9))
    gs = fig_comb.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.2)

    # (a)
    ax1 = fig_comb.add_subplot(gs[0, 0])
    img1 = librosa.display.specshow(plot_S_deg, sr=sr, hop_length=plot_hop_length, x_axis='time', y_axis='hz', ax=ax1, cmap='magma')
    ax1.plot(plot_times, plot_cutoffs, color='cyan', linewidth=1.5, linestyle='--', label='Detected Cutoff')
    ax1.set_title(f'(a) Input Spectrogram (LP: {eval_cutoff}Hz)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    fig_comb.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # (b)
    ax2 = fig_comb.add_subplot(gs[0, 1])
    img2 = librosa.display.specshow(plot_S_enh, sr=sr, hop_length=plot_hop_length, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title('(b) Restored Spectrogram', fontweight='bold')
    ax2.set_ylabel('') 
    fig_comb.colorbar(img2, ax=ax2, format='%+2.0f dB')

    # (c)
    ax3 = fig_comb.add_subplot(gs[1, 0])
    ax3.plot(plot_times, plot_tilts, color='#6a0dad', linewidth=1.5)
    ax3.set_title('(c) Spectral Tilt Evolution', fontweight='bold')
    ax3.set_ylabel('Tilt (dB)')
    ax3.set_ylim(-50, 5)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # (d) Combined Version
    ax4 = fig_comb.add_subplot(gs[1, 1])
    ax4.semilogx(fft_freqs, frame_db, color='gray', alpha=0.5, label='Raw Spectrum')
    if cutoff_idx > low_cut_idx + 2 and np.sum(mask) > 1:
        ax4.semilogx(x_plot, y_plot, color='red', linewidth=2.0, label='Regression Fit')
        ax4.semilogx(x_ext, y_ext, color='red', linewidth=1.0, linestyle=':', alpha=0.7)
        ax4.text(0.05, 0.1, f'Tilt: {calc_tilt:.1f}dB', transform=ax4.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    ax4.axvline(x=GAIN_CALC_LOW_CUT_HZ, color='green', linestyle=':', label='Fit Start')
    ax4.axvline(x=center_cutoff, color='blue', linestyle=':', label='Cutoff')
    
    ax4.set_xscale('log')
    ax4.set_xticks([100, 500, 1000, 5000, 10000, 20000])
    ax4.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax4.get_xaxis().set_minor_formatter(ticker.NullFormatter())

    ax4.set_title(f'(d) Regression @ Center ({times[center_frame_idx_raw]:.1f}s)', fontweight='bold')
    ax4.set_ylabel('Mag (dB)')
    ax4.set_xlim(100, sr/2)
    ax4.set_ylim(-80, 5)
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend(loc='upper right', fontsize=9)

    # (e)
    ax5 = fig_comb.add_subplot(gs[2, :])
    ax5.plot(f, 10*np.log10(P_orig+1e-12), color='green', alpha=0.4, label='Original', linewidth=1.0)
    ax5.plot(f, 10*np.log10(P_deg+1e-12), color='blue', alpha=0.6, label='Degraded', linewidth=1.0)
    ax5.plot(f, 10*np.log10(P_sh+1e-12), color='orange', linestyle='--', label='S&H Raw', linewidth=1.0, alpha=0.7)
    ax5.plot(f, 10*np.log10(P_enh+1e-12), color='red', label='Restored', linewidth=1.5)
    ax5.set_title('(e) Overall PSD', fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('PSD (dB/Hz)')
    ax5.set_xlim(0, 24000)
    ax5.set_ylim(bottom=-100)
    ax5.legend(loc='lower left', ncol=4, fontsize=9)
    ax5.grid(True, which='both', alpha=0.3)

    save_individual_plot(fig_comb, out_filename, "_plot_Combined_Summary")

# ==========================================
# 処理パイプライン
# ==========================================

def process_single_pass(y_input, sr, eval_cutoff_freq=None):
    if eval_cutoff_freq is not None:
        print(f"\n[Eval Mode] Applying Low-pass filter at {eval_cutoff_freq} Hz...")
        y_process = apply_lowpass_filter(y_input, sr, eval_cutoff_freq)
        degraded_signal = y_process.copy()
    else:
        y_process = y_input
        degraded_signal = y_input 

    num_channels = 1 if y_process.ndim == 1 else y_process.shape[1]
    
    print(" -> Analyzing signal structure (Method 3: Energy Scan)...")
    y_ana = y_process if num_channels == 1 else y_process[:, 0]
    times, detected_cutoffs, calculated_tilts = detect_cutoff_and_slope(y_ana, sr)

    print(" -> Interpolating parameters...")
    f_cutoff = interp1d(times, detected_cutoffs, kind='linear', fill_value="extrapolate")
    f_tilt = interp1d(times, calculated_tilts, kind='linear', fill_value="extrapolate")
    
    samples_len = len(y_process)
    audio_duration = samples_len / sr
    sample_indices = np.linspace(0, audio_duration, samples_len)
    
    cutoffs_sample = f_cutoff(sample_indices).astype(np.float64)
    tilts_sample = f_tilt(sample_indices).astype(np.float64)
    
    print(" -> Processing with Sample & Hold + Dual Shelf Filter...")
    if num_channels == 1:
        enhanced, sh_raw = apply_dynamic_processing(y_process.astype(np.float64), cutoffs_sample, tilts_sample, float(sr))
    else:
        enhanced = np.zeros_like(y_process)
        sh_raw = np.zeros_like(y_process)
        for ch in range(num_channels):
            e_ch, sh_ch = apply_dynamic_processing(y_process[:, ch].astype(np.float64), cutoffs_sample, tilts_sample, float(sr))
            enhanced[:, ch] = e_ch
            sh_raw[:, ch] = sh_ch
            
    return degraded_signal, sh_raw, enhanced, times, detected_cutoffs, calculated_tilts

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Dynamic SBR (Log Axis + Tick Fix + Memory Safe)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input audio path')
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_lowpass', type=int, default=None, help='Specific cutoff freq')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()

    try:
        y, sr = librosa.load(args.input, sr=ORIGINAL_SR, mono=False)
        if y.ndim > 1: y = y.T
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    if args.eval:
        print("=== EVALUATION MODE STARTED ===")
        if args.eval_lowpass is not None:
            test_cutoffs = [args.eval_lowpass]
        else:
            test_cutoffs = [
                random.randint(3500, 5500),   
                random.randint(6000, 9000),   
                random.randint(10000, 14000)  
            ]
            print(f"Randomly selected cutoffs: {test_cutoffs}")

        for cutoff in test_cutoffs:
            print(f"\n--- Testing with Cutoff: {cutoff} Hz ---")
            deg_y, sh_y, enh_y, times, d_cutoffs, d_tilts = process_single_pass(y, sr, eval_cutoff_freq=cutoff)
            
            out_enh = f"{base_name}_eval_{cutoff}Hz_enhanced.wav"
            out_deg = f"{base_name}_eval_{cutoff}Hz_degraded.wav"
            out_sh  = f"{base_name}_eval_{cutoff}Hz_sh_raw.wav"

            sf.write(out_enh, enh_y, sr)
            sf.write(out_deg, deg_y, sr)
            sf.write(out_sh, sh_y, sr)
            print(f"Saved audio files.")

            mse_sh = np.mean((y - sh_y) ** 2)
            mse_enh = np.mean((y - enh_y) ** 2)
            print(f"MSE (Original vs S&H Raw):   {mse_sh:.6e}")
            print(f"MSE (Original vs Enhanced):  {mse_enh:.6e}")

            if not args.no_plot:
                print("Generating plots (Memory Safe Mode)...")
                plot_paper_evaluation(y, deg_y, sh_y, enh_y, sr, times, d_cutoffs, d_tilts, cutoff, out_enh)

    else:
        print("=== NORMAL PROCESSING MODE ===")
        _, sh_y, enh_y, times, d_cutoffs, d_tilts = process_single_pass(y, sr, eval_cutoff_freq=None)
        
        out_name = f"{base_name}_enhanced.wav"
        sf.write(out_name, enh_y, sr)
        print(f"Saved: {out_name}")

        mse_sh = np.mean((y - sh_y) ** 2)
        mse_enh = np.mean((y - enh_y) ** 2)
        print(f"MSE (Original vs S&H Raw):   {mse_sh:.6e}")
        print(f"MSE (Original vs Enhanced):  {mse_enh:.6e}")

        if not args.no_plot:
            print("Generating plots (Memory Safe Mode)...")
            plot_paper_evaluation(y, y, sh_y, enh_y, sr, times, d_cutoffs, d_tilts, "None (Native)", out_name)

if __name__ == '__main__':
    main()