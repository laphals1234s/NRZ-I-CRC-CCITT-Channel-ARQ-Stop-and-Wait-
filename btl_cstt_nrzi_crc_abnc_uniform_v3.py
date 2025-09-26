# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================
# Text <-> Bits (ASCII 8-bit)
# ============================================
def text_to_bits_ascii8(s: str):
    data = s.encode('latin-1', errors='ignore')  # enforce 0..255 per char
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def bits_to_text_ascii8(bits):
    out = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | (bits[i + j] & 1)
            out.append(byte)
    try:
        return bytes(out).decode('latin-1', errors='ignore')
    except Exception:
        return ''.join(chr(b) for b in out)

# ============================================
# CRC-CCITT (x^16 + x^12 + x^5 + 1), non-reflected, init 0x0000, no final XOR
# ============================================
CRC_POLY = 0x1021
CRC_WIDTH = 16

def _msb_pos(x: int) -> int:
    return -1 if x == 0 else (x.bit_length() - 1)

def crc_ccitt_bits(msg_bits):
    value = 0
    for b in msg_bits:
        value = (value << 1) | (b & 1)
    value <<= CRC_WIDTH
    poly = (1<<CRC_WIDTH) | CRC_POLY
    work = value
    poly_msb = _msb_pos(poly)
    while True:
        w_msb = _msb_pos(work)
        if w_msb < poly_msb:
            break
        shift = w_msb - poly_msb
        work ^= (poly << shift)
    rem_bits = [(work >> i) & 1 for i in range(CRC_WIDTH - 1, -1, -1)]
    return rem_bits

def append_crc(msg_bits):
    R = crc_ccitt_bits(msg_bits)
    return msg_bits + R, R

def crc_check_full(rx_bits):
    val = 0
    for b in rx_bits:
        val = (val << 1) | (b & 1)
    poly = (1<<CRC_WIDTH) | CRC_POLY
    work = val
    poly_msb = _msb_pos(poly)
    while True:
        w_msb = _msb_pos(work)
        if w_msb < poly_msb:
            break
        shift = w_msb - poly_msb
        work ^= (poly << shift)
    rem_bits = [(work >> i) & 1 for i in range(CRC_WIDTH - 1, -1, -1)]
    return rem_bits

# ============================================
# NRZ-I
# 0 -> invert level, 1 -> hold
# ============================================
def nrzi_encode(bits, V_low, V_high, initial_low=True):
    cur = V_low if initial_low else V_high
    y = []
    for b in bits:
        if b == 0:
            cur = V_high if np.isclose(cur, V_low) else V_low
        y.append(cur)
    return np.array(y, dtype=float)

def sample_centers(y: np.ndarray, samples_per_bit: int) -> np.ndarray:
    if samples_per_bit <= 1:
        return y
    n_bits = len(y) // samples_per_bit
    c = samples_per_bit // 2
    idx = np.arange(n_bits) * samples_per_bit + c
    return y[idx]

def nrzi_decode_from_samples(symbol_samples, initial_low=True, V_low=None, V_high=None):
    thr = 0.5 * (V_low + V_high) if (V_low is not None and V_high is not None) else float(np.median(symbol_samples))
    levels = np.where(symbol_samples >= thr, 1, -1)
    bits = []
    prev = -1 if initial_low else 1
    for lv in levels:
        bits.append(0 if lv != prev else 1)
        prev = lv
    return bits

# ============================================
# Channel: ABNC (Additive Bounded-Noise Channel) — Uniform noise
#   y = x + n, with n ~ Uniform[-N, N]  (|n| ≤ N).
#   SNR (dB) = 20*log10(Vs/N),  Vs = V_high - V_low.
# ============================================
def abnc(signal, snr_db, V_low, V_high):
    x = signal.astype(float)
    Vs = float(V_high - V_low)
    N = Vs / (10.0 ** (snr_db / 20.0)) if Vs != 0 else 0.0
    noise = np.random.uniform(-N, N, size=x.shape)
    return x + noise, noise, N

# ============================================
# Helpers
# ============================================
def plot_stairs(ax, values, title, ylabel, xlabel="Sample"):
    ax.step(np.arange(len(values)), values, where='post')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True)

def bits_to_hex(bits):
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return ''.join(f"{b:02X}" for b in out)

def diff_highlight_html(tx_bits, rx_bits):
    L = min(len(tx_bits), len(rx_bits))
    segs = []
    for i in range(L):
        btx, brx = tx_bits[i], rx_bits[i]
        if btx != brx:
            segs.append(f"<mark style='background-color:#ffdf80'>{brx}</mark>")
        else:
            segs.append(str(brx))
    if len(rx_bits) > L:
        segs.append(''.join(str(b) for b in rx_bits[L:]))
    return ''.join(segs)

# ============================================
# App (V12 classic, ABNC, no ARQ)
# ============================================
st.set_page_config(page_title="NRZ-I + CRC — V12 (classic, ABNC)", layout="wide")
st.title("NRZ-I + CRC-CCITT + Channel — V12 (classic UI, ABNC)")
st.caption("Một lần truyền; kiểm tra CRC để xác nhận khung.")

with st.sidebar:
    st.header("Input Data")
    text = st.text_input("Nhập 6 ký tự ASCII (8-bit/char)", value="VNPT24")
    # Strict ASCII printable, exactly 6
    if len(text) != 6 or any(ord(ch) < 0x20 or ord(ch) > 0x7E for ch in text):
        st.error("Yêu cầu đúng **6 ký tự ASCII hiển thị** (0x20–0x7E).")
        st.stop()
    initial_low = st.checkbox("Mức ban đầu = V_low (NRZ-I)", value=True)
    V_low_str  = st.text_input("V_low (V)",  value="2,0")
    V_high_str = st.text_input("V_high (V)", value="7,0")

    # Parse x,x with comma or dot; enforce one decimal place
    def _parse_one_decimal(s, name):
        s0 = (s or "").strip().replace(",", ".")
        try:
            val = float(s0)
        except Exception:
            st.error(f"{name}: nhập dạng x, x (ví dụ 5,5).")
            st.stop()
        return round(val, 1)

    V_low = _parse_one_decimal(V_low_str, "V_low (V)")
    V_high = _parse_one_decimal(V_high_str, "V_high (V)")
    if not (V_low < V_high):
        st.error("Yêu cầu: V_high phải > V_low.")
        st.stop()
    samples_per_bit = st.number_input("Samples per bit (để vẽ chi tiết)", min_value=1, value=10, step=1)

    st.header("CRC")
    st.write("CRC-CCITT (x^16 + x^12 + x^5 + 1), non-reflected, init=0x0000, no final XOR.")

    st.header("Channel")
    ch_type = st.selectbox("Loại kênh", ["ABNC (Uniform: n[k] ∈ [-N, N])"])

    # SNR text input: default 12,04; parse to 2 decimals; clamp [0, 40].
    snr_str = st.text_input("SNR (dB)", value="12,04", help="Nhập số, 2 chữ số sau dấu phẩy. Giới hạn 0–40 dB. Ấn Enter để áp dụng.")
    def _parse_two_decimals(s, name):
        s0 = (s or "").strip().replace(",", ".")
        try:
            val = float(s0)
        except Exception:
            st.error(f"{name}: chỉ cho phép số (ví dụ 12,04).")
            st.stop()
        # clamp and round to 2 decimals
        val = max(0.0, min(40.0, val))
        return float(f"{val:.2f}")
    snr_db = _parse_two_decimals(snr_str, "SNR (dB)")

    Vs_tmp = (V_high - V_low)
    N_bound = Vs_tmp / (10**(snr_db/20)) if Vs_tmp != 0 else 0.0
    st.caption("**Additive Bounded-Noise Channel - ABNC (Uniform: n[k] ∈ [-N, N] — y[k] = x[k] + n[k],  N = Vs/10^(SNR/20))**")

# Build payload & append CRC
M_bits = text_to_bits_ascii8(text)
M_len = len(M_bits)
tx_bits, fcs_bits = append_crc(M_bits)
frame_len = len(tx_bits)

# NRZ-I symbols and upsampling for display
def upsample_stairs(symbols, k):
    return symbols if k <= 1 else np.repeat(symbols, k)

sym = nrzi_encode(tx_bits, V_low=V_low, V_high=V_high, initial_low=initial_low)
y_tx = upsample_stairs(sym, samples_per_bit)

# ---------- Channel forward pass (Single Try) ----------
def abnc_once(y_tx, snr_db, V_low, V_high):
    y_rx, noise, N = abnc(y_tx, snr_db=snr_db, V_low=V_low, V_high=V_high)
    return y_rx, noise, N

y_rx, noise, N = abnc_once(y_tx, snr_db=snr_db, V_low=V_low, V_high=V_high)

# Symbol-rate samples at centers of each bit
rx_symbol_samples = sample_centers(y_rx, samples_per_bit)
rx_bits = nrzi_decode_from_samples(rx_symbol_samples, initial_low=initial_low, V_low=V_low, V_high=V_high)

# Threshold & margins
Vs = float(V_high - V_low)
thr = 0.5 * (V_high + V_low)
Nmax = abs(V_high - V_low) / 2.0
margin_V = Nmax - N
margin_dB = snr_db - 6.02

# BER from actual decode
comp_len = min(len(tx_bits), len(rx_bits))
bit_errors = sum(1 for i in range(comp_len) if tx_bits[i] != rx_bits[i])
ber = bit_errors / comp_len if comp_len > 0 else 0.0

# CRC check on received (full frame length)
rem = crc_check_full(rx_bits[:frame_len])
crc_ok = all(x == 0 for x in rem)

# Recovered text
rx_text = bits_to_text_ascii8(rx_bits[:M_len])

# =============== Layout outputs ===============
st.markdown("## Kết quả khung hiện tại (1 lần truyền)")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Độ dài payload (bit)", M_len)
    st.metric("Độ dài khung (bit) M||FCS", frame_len)
with c2:
    st.metric("Số bit lỗi (so với TX)", bit_errors)
    st.metric("BER", f"{ber:.6f}")
with c3:
    st.metric("CRC check", "PASS" if crc_ok else "FAIL")
    st.metric("Ngưỡng tách mức (V_th)", f"{thr:.3f} V")

st.markdown("### Thông số kênh & biên an toàn")
c4, c5, c6 = st.columns(3)
with c4:
    st.metric("V_low / V_high (V)", f"{V_low:.3f} / {V_high:.3f}")
    st.metric("Nmax (|Vh-Vl|/2)", f"{Nmax:.3f} V")
with c5:
    st.metric("SNR (dB)", f"{snr_db:.2f}")
    st.metric("N (biên nhiễu, V)", f"{N:.4f}")
with c6:
    st.metric("Margin (V)", f"{margin_V:.4f}")
    st.metric("Margin (dB)", f"{margin_dB:.2f}")

# Plots
st.markdown("### Biểu diễn dạng sóng (TX/RX)")
# Use constrained_layout to auto-manage spacing; legend INSIDE lower-right (Method A)
fig, axes = plt.subplots(2, 1, figsize=(12, 5), constrained_layout=True)

# Draw V_th in consistent green
axes[0].axhline(thr, linestyle='--', linewidth=0.9, color='tab:green', label="V_th")
plot_stairs(axes[0], y_tx, "TX NRZ-I waveform (upsampled)", "Voltage (V)")
axes[1].axhline(thr, linestyle='--', linewidth=0.9, color='tab:green', label="V_th")
plot_stairs(axes[1], y_rx, "RX waveform (ABNC Uniform)", "Voltage (V)")

# Legend for each axis: put to the top-right (same line with title), outside plotting area
for ax in axes:
    leg = ax.legend(loc='lower right',
                    bbox_to_anchor=(1.0, 1.02),  # above the axes, right side
                    ncol=1, frameon=True, framealpha=0.95,
                    facecolor='white', edgecolor='0.7',
                    fontsize=10, handlelength=1.4,
                    borderpad=0.25, labelspacing=0.25)
    try:
        leg.get_frame().set_linewidth(0.8)
    except Exception:
        pass

st.pyplot(fig)
# ------------------------------------------------------------------
# NEW: 3-panel figure (TX | ±N band & V_th | RX)  -- keep all else same
# ------------------------------------------------------------------
st.markdown("### TX / ±N / RX (3-panel, cùng trục thời gian)")
try:
    x_idx = np.arange(len(y_tx))
    upper = y_tx + N
    lower = y_tx - N
    # vùng có thể lật bit khi dải ±N cắt ngưỡng
    unsafe = (lower <= thr) & (upper >= thr)

    fig3, ax = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)

    # Panel 1: TX NRZ-I
    plot_stairs(ax[0], y_tx, "TX NRZ-I", "Voltage (V)")
    ax[0].axhline(thr, linestyle='--', linewidth=0.9, color='g', label="V_th")
    # Legend top-right above axis to avoid covering title
    leg0 = ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=1,
                        frameon=True, framealpha=0.95)

    # Panel 2: ±N band around TX + highlight unsafe
    ax[1].step(x_idx, y_tx, where='post', linewidth=1.2, label="TX")
    ax[1].fill_between(x_idx, lower, upper, step='post', alpha=0.25, label=f"±N ({N:.2f} V)")
    ax[1].fill_between(x_idx, lower, upper, where=unsafe, step='post', alpha=0.35, label="Vùng có thể lật bit")
    ax[1].axhline(thr, linestyle='--', linewidth=0.9, color='g', label="V_th")
    ax[1].set_ylabel("Voltage (V)")
    ax[1].set_title("Dải nhiễu ±N quanh TX (ABNC)")

    # Hai cụm chú giải: (TX, ±N) bên trái; (Vùng có thể lật bit, V_th) bên phải; nằm trên 1 hàng ngang
    handles, labels = ax[1].get_legend_handles_labels()
    left_h = []; left_l = []; right_h = []; right_l = []
    for h, l in zip(handles, labels):
        if l.startswith("TX") or l.startswith("±N"):
            left_h.append(h); left_l.append(l)
        else:
            right_h.append(h); right_l.append(l)
    leg_left = ax[1].legend(left_h, left_l, loc="lower left",
                            bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=True, framealpha=0.95)
    ax[1].add_artist(leg_left)
    ax[1].legend(right_h, right_l, loc="lower right",
                 bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=True, framealpha=0.95)

    # Panel 3: RX = TX + n(t)
    plot_stairs(ax[2], y_rx, "RX waveform (ABNC Uniform)", "Voltage (V)", xlabel="Sample")
    ax[2].axhline(thr, linestyle='--', linewidth=0.9, color='g', label="V_th")
    # Legend top-right above axis
    leg2 = ax[2].legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=1,
                        frameon=True, framealpha=0.95)

    # Đồng bộ thang điện áp 3 panel để so sánh trực quan
    vmin = float(min(np.min(lower), np.min(y_rx)))
    vmax = float(max(np.max(upper), np.max(y_rx)))
    pad  = 0.02 * max(1.0, (vmax - vmin))
    for a in ax:
        a.set_ylim(vmin - pad, vmax + pad)
        a.grid(True)

    # Nhãn dưới cùng: hiển thị so sánh N với N_max, gồm cả trường hợp bằng
    Nmax = abs(V_high - V_low) / 2.0
    if N < Nmax:
        arrow_lbl = "→  N < N_max"
    elif N > Nmax:
        arrow_lbl = "→  N > N_max"
    else:
        arrow_lbl = "→  N = N_max"
    ax[2].text(0.01, -0.28,
               f"N = {N:.3f} V,  N_max = {Nmax:.3f} V,  margin = {Nmax - N:.3f} V  {arrow_lbl}",
               transform=ax[2].transAxes, ha="left", va="top")

    st.pyplot(fig3)
except Exception as _e:
    st.info("Không thể vẽ 3-panel TX/±N/RX (thiếu biến).")
# ------------------------------------------------------------------



# --- TX vs RX (symbol-rate) overlay ---
st.markdown("### So sánh TX symbols vs RX samples (symbol-rate)")
try:
    fig_cmp, ax_cmp = plt.subplots(1, 1, figsize=(10, 3.6), constrained_layout=False)
    x = np.arange(len(sym))
    ax_cmp.step(x, sym, where='post', label="TX symbols")
    ax_cmp.step(x, rx_symbol_samples, where='post', label="RX samples")
    ax_cmp.axhline(thr, linestyle="--", linewidth=1, color="tab:green", label=f"V_th={thr:.3f} V")
    ax_cmp.set_ylabel("Voltage (V)")
    ax_cmp.set_xlabel("Bit index k (0-based)")
    ax_cmp.set_title("TX vs RX @ symbol-rate (k)")
    from matplotlib.ticker import MultipleLocator
    ax_cmp.set_xlim(0, max(0, len(sym)-1))
    ax_cmp.xaxis.set_major_locator(MultipleLocator(4))
    ax_cmp.xaxis.set_minor_locator(MultipleLocator(1))
    ax_cmp.grid(True, which="major", axis="x", linestyle="-", linewidth=0.6, alpha=0.35)
    ax_cmp.grid(True, which="minor", axis="x", linestyle="--", linewidth=0.8, alpha=0.75)
    ax_cmp.tick_params(axis="x", labelsize=9)
    handles, labels = ax_cmp.get_legend_handles_labels()
    ax_cmp.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=True)
    fig_cmp.subplots_adjust(top=0.78)
    ax_cmp.grid(True)
    st.pyplot(fig_cmp)
except Exception as _e:
    st.info("Không thể vẽ overlay TX/RX ở symbol-rate (thiếu biến).")

# ============================================
# Detailed frame info (TX vs RX) + highlight
# ============================================
with st.expander("Chi tiết khung & CRC (TX vs RX + highlight lỗi)"):
    st.write(f"Payload bits (M): {M_len} bit")
    st.code(''.join(str(b) for b in M_bits), language='text')
    st.write(f"FCS (16 bit, TX) = 0x{int(''.join(str(b) for b in fcs_bits), 2):04X}")
    st.code(''.join(str(b) for b in fcs_bits), language='text')
    st.write(f"Frame TX (M||FCS): len={frame_len} bit; HEX = {bits_to_hex(tx_bits)}")

    st.markdown('---')
    st.write("**Frame RX (M||FCS) — đã giải NRZ-I, trước khi kiểm CRC**")
    rx_bits_view = rx_bits[:frame_len]
    st.code(''.join(str(b) for b in rx_bits_view), language='text')

    html = diff_highlight_html(tx_bits, rx_bits_view)
    st.markdown("**RX bits (highlight khác TX):**", unsafe_allow_html=True)
    st.markdown(f"<pre style='white-space:pre-wrap'>{html}</pre>", unsafe_allow_html=True)

    err_idx0 = [i for i in range(min(len(tx_bits), len(rx_bits_view))) if tx_bits[i] != rx_bits_view[i]]
    if err_idx0:
        err_idx1 = [i+1 for i in err_idx0]
        st.write(f"**Vị trí bit sai (1-based):** {err_idx1}")
    else:
        st.write("**Vị trí bit sai (1-based):** —")

    rem_bits = crc_check_full(rx_bits_view)
    st.write(f"Remainder (RX / G(x)) = {''.join(str(b) for b in rem_bits)} → {'PASS' if all(x==0 for x in rem_bits) else 'FAIL'}")

# ============================================
# Process flow sketch (ASCII diagram)
# ============================================
with st.expander("Sơ đồ tóm tắt tiến trình xử lý"):
    flow = r"""
Text (6 chars)
       ↓  ASCII (8-bit)
  Chuỗi bit M
       ↓  CRC-CCITT (x^16 + x^12 + x^5 + 1)
  Khung TX = M || FCS (16)
       ↓  NRZ-I Encoder (0→đảo, 1→giữ), V_low/V_high
  Dạng sóng mức (TX)
       ↓  Kênh (ABNC Uniform: n[k] ∈ [-N, N]), SNR = 20·log10(Vs/N)
  Dạng sóng mức (RX)
       ↓  Lấy mẫu @ center & Ngưỡng V_th
  Ký hiệu RX (symbol-rate)
       ↓  NRZ-I Decoder (so sánh chuyển mức)
  Chuỗi bit RX
       ↓  CRC check (RX / G(x))
"""
    st.code(flow, language="text")

# Bottom note
st.caption("Ghi chú: NRZ-I không có cơ chế 'phát hiện lỗi ở lớp mã đường' kiểu vi phạm (như AMI/HDB3). Do đó trường hợp (b) trong đề không áp dụng trực tiếp cho NRZ-I; các lỗi được phát hiện nhờ CRC ở lớp kiểm soát lỗi dữ liệu.")
