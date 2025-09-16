
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================
# Utility: Text <-> Bits (ASCII 8-bit)
# ============================================
def text_to_bits_ascii8(s: str):
    """Return list[int] bits for exactly len(s)*8 bits using ASCII/Latin-1 (0..255)."""
    data = s.encode('latin-1', errors='ignore')  # enforce 0..255 per char
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def bits_to_text_ascii8(bits):
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        if i + 8 <= len(bits):
            for j in range(8):
                byte = (byte << 1) | (bits[i + j] & 1)
            out.append(byte)
    try:
        return bytes(out).decode('latin-1', errors='ignore')
    except Exception:
        return ''.join(chr(b) for b in out)

# ============================================
# CRC-CCITT (x^16 + x^12 + x^5 + 1) - non-reflected, init 0x0000, no final XOR
# ============================================
CRC_POLY = 0x1021  # x^16 + x^12 + x^5 + 1
CRC_WIDTH = 16

def _msb_pos(x: int) -> int:
    if x == 0:
        return -1
    return x.bit_length() - 1

def crc_ccitt_bits(msg_bits):
    """Return remainder bits (length 16) via polynomial long division (bit-aligned)."""
    value = 0
    for b in msg_bits:
        value = (value << 1) | (b & 1)
    value <<= CRC_WIDTH  # append r zeros
    poly = CRC_POLY
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
    """Return remainder of dividing rx_bits by generator; all zeros => pass."""
    val = 0
    for b in rx_bits:
        val = (val << 1) | (b & 1)
    poly = CRC_POLY
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
# NRZ-I encode/decode
# Rule: bit 0 -> invert level; bit 1 -> hold level (differential)
# Levels: V_low, V_high. initial_low=True means initial level is V_low else V_high.
# ============================================
def nrzi_encode(bits, V_low, V_high, initial_low=True):
    cur = V_low if initial_low else V_high
    y = []
    for b in bits:
        if b == 0:  # transition
            cur = V_high if np.isclose(cur, V_low) else V_low
        y.append(cur)
    return np.array(y, dtype=float)

def nrzi_decode(symbol_samples, initial_low=True, V_low=None, V_high=None):
    """Decode NRZI from symbol-rate samples (one sample per bit).
       Threshold = mid between V_low/V_high if provided; else median.
       Transition => bit 0, No-transition => bit 1.
    """
    if V_low is not None and V_high is not None:
        thr = 0.5 * (V_low + V_high)
    else:
        thr = np.median(symbol_samples)
    # map to -1/+1 for robust transition detection
    levels = np.where(symbol_samples >= thr, 1, -1)
    bits = []
    prev = -1 if initial_low else 1
    for lv in levels:
        bits.append(0 if lv != prev else 1)
        prev = lv
    return bits

# ============================================
# Channel models
# ============================================
def awgn(signal, snr_db):
    """AWGN for given SNR(dB) measured against signal AC power around mean."""
    x = signal.astype(float)
    mu = np.mean(x)
    sig_ac = x - mu
    Ps = np.mean(sig_ac**2) + 1e-18  # signal power
    snr_lin = 10 ** (snr_db / 10.0)
    Pn = Ps / snr_lin
    noise = np.random.normal(0.0, np.sqrt(Pn), size=x.shape)
    return x + noise, noise, Pn

def apply_awgn_with_noise(signal, noise):
    return signal.astype(float) + noise

def bsc_flip_bits(bits, p):
    """Binary Symmetric Channel flips with prob p on bit domain."""
    flips = np.random.rand(len(bits)) < p
    out = [(b ^ 1) if f else b for b, f in zip(bits, flips)]
    return out, flips

def apply_bsc_with_flips(bits, flips):
    return [(b ^ 1) if f else b for b, f in zip(bits, flips)]

# ============================================
# Plot helpers
# ============================================
def plot_stairs(ax, values, title, ylabel, xlabel="Sample"):
    ax.step(np.arange(len(values)), values, where='post')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True)

def plot_ack_trace(ax, trace):
    ax.step(np.arange(1, len(trace) + 1), trace, where='post')
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['NAK', 'ACK'])
    ax.set_xlabel("Attempt #")
    ax.set_title("ACK/NAK by Attempt")
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
    """Return an HTML string with differing bits wrapped in <mark> (RX view)."""
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
# Main App
# ============================================
st.set_page_config(page_title="NRZ-I + CRC + ARQ Simulator (v3)", layout="wide")

st.title("NRZ-I + CRC-CCITT + Channel + ARQ (Stop-and-Wait) — v3")
st.caption("Attempt #1 trong ARQ dùng đúng mẫu nhiễu với phần '1 lần truyền' để kết quả nhất quán.")

with st.sidebar:
    st.header("Input Data")
    text = st.text_input("Nhập 6 ký tự ASCII (8-bit/char)", value="VNPT24")[:6]
    if len(text) < 6:
        st.warning("Bạn nên nhập đúng 6 ký tự ASCII (mỗi ký tự 8 bit).")
    initial_low = st.checkbox("Mức ban đầu = V_low (NRZ-I)", value=True)
    V_low = st.number_input("V_low (V)", value=2.0, step=0.1)
    V_high = st.number_input("V_high (V)", value=7.0, step=0.1)
    samples_per_bit = st.number_input("Samples per bit (để vẽ chi tiết)", min_value=1, value=10, step=1)

    st.header("CRC")
    st.write("CRC-CCITT (x^16 + x^12 + x^5 + 1), non-reflected, init=0x0000, no final XOR.")

    st.header("Channel")
    ch_type = st.selectbox("Loại kênh", ["AWGN (SNR dB)", "Bit-Flip (BSC)"])
    if ch_type == "AWGN (SNR dB)":
        snr_db = st.slider("SNR (dB)", -5, 30, 12, 1)
        st.caption("Lưu ý: Công suất tín hiệu S được chuẩn hóa = 1 (theo AC power). Khi thay đổi SNR, nhiễu N được tính theo N = S / 10^(SNR/10).")
    else:
        p_flip = st.slider("Xác suất lật bit p", 0.0, 0.2, 0.02, 0.005)
        st.caption("BSC coi mỗi bit bị lật độc lập với xác suất p.")

    st.header("ARQ (ACK/NAK)")
    max_retx = st.slider("Số lần phát lại tối đa", 1, 10, 5, 1)

# Build payload & append CRC
M_bits = text_to_bits_ascii8(text)
M_len = len(M_bits)  # payload length
tx_bits, fcs_bits = append_crc(M_bits)
frame_len = len(tx_bits)

# NRZ-I symbols and upsampling for display
def upsample_stairs(symbols, k):
    if k <= 1:
        return symbols
    return np.repeat(symbols, k)

sym = nrzi_encode(tx_bits, V_low=V_low, V_high=V_high, initial_low=initial_low)
y_tx = upsample_stairs(sym, samples_per_bit)

# =====================================
# Channel forward pass (Single Try) — store the EXACT channel realization
# =====================================
shared_noise = None
shared_flips = None

if ch_type == "AWGN (SNR dB)":
    y_rx, noise, Pn = awgn(y_tx, snr_db=snr_db)  # noise realization
    shared_noise = noise.copy()
    # symbol-rate sampling
    rx_symbol_samples = y_rx[samples_per_bit-1::samples_per_bit] if samples_per_bit > 1 else y_rx
    rx_bits = nrzi_decode(rx_symbol_samples, initial_low=initial_low, V_low=V_low, V_high=V_high)
    # Noise metrics
    N_rms = float(np.sqrt(np.mean((noise**2)))) if noise is not None else 0.0
    N_peak = float(np.max(np.abs(noise))) if noise is not None else 0.0
else:
    rx_bits_tmp, flips = bsc_flip_bits(tx_bits, p_flip)
    shared_flips = flips.copy()
    rx_bits = rx_bits_tmp
    y_rx = upsample_stairs(nrzi_encode(rx_bits, V_low=V_low, V_high=V_high, initial_low=initial_low), samples_per_bit)
    Pn = None
    N_rms = None
    N_peak = None

# BER
comp_len = min(len(tx_bits), len(rx_bits))
bit_errors = sum(1 for i in range(comp_len) if tx_bits[i] != rx_bits[i])
ber = bit_errors / comp_len if comp_len > 0 else 0.0

# CRC check on received (full frame length)
rem = crc_check_full(rx_bits[:frame_len])
crc_ok = all(x == 0 for x in rem)

# Recovered text
rx_text = bits_to_text_ascii8(rx_bits[:M_len])

# Threshold & Nmax
Nmax = abs(V_high - V_low) / 2.0
thr = 0.5 * (V_high + V_low)

# Case classifier (for requirement 3a/3c; 3b does not apply to NRZ-I inherently)
if bit_errors == 0:
    case_label = "A"
elif not crc_ok:
    case_label = "C"
else:
    case_label = "A"  # rare undetected errors by CRC would show as 'A' here

# ===============
# Layout outputs
# ===============
st.markdown("## Kết quả khung hiện tại (1 lần truyền)")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Độ dài payload (bit)", M_len)
    st.metric("Độ dài khung (bit) M||FCS", frame_len)
with c2:
    st.metric("Số bit lỗi (so với TX)", bit_errors)
    st.metric("BER", f"{ber:.6f}")
with c3:
    st.metric("CRC check", "PASS ✅" if crc_ok else "FAIL ❌")
    st.metric("Ngưỡng tách mức (V_th)", f"{thr:.3f} V")

st.markdown("### Thông số định lượng tín hiệu & kênh")
c4, c5, c6 = st.columns(3)
with c4:
    st.metric("V_low / V_high (V)", f"{V_low:.3f} / {V_high:.3f}")
    st.metric("Nmax (|Vh-Vl|/2)", f"{Nmax:.3f} V")
with c5:
    if ch_type == "AWGN (SNR dB)":
        st.metric("SNR đặt (dB)", f"{snr_db:.2f}")
        mu_tx = np.mean(y_tx)
        Ps = np.mean((y_tx - mu_tx) ** 2) + 1e-18
        Pn_emp = np.mean((y_rx - y_tx) ** 2) + 1e-18
        snr_emp = 10 * np.log10(Ps / Pn_emp)
        st.metric("SNR thực nghiệm (dB)", f"{snr_emp:.2f}")
    else:
        st.metric("Flip prob p", f"{p_flip:.4f}")
with c6:
    st.metric("Payload (ASCII)", text)
    st.metric("Thu lại (best-effort)", rx_text if len(rx_text) == 6 else (rx_text + " " * (6 - len(rx_text))))
if ch_type == "AWGN (SNR dB)":
    c7, c8 = st.columns(2)
    with c7:
        st.metric("N_rms (V)", f"{N_rms:.6f}")
    with c8:
        st.metric("N_peak quan sát (V)", f"{N_peak:.6f}")

# Plots
st.markdown("### Biểu diễn dạng sóng (TX/RX)")
fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
plot_stairs(axes[0], y_tx, "TX NRZ-I waveform (upsampled)", "Voltage (V)")
plot_stairs(axes[1], y_rx, "RX waveform", "Voltage (V)")
st.pyplot(fig)

# ============================================
# ARQ (Stop-and-Wait) — Attempt #1 uses the SAME channel realization
# ============================================
st.markdown("---")
st.header("🔄 ARQ (ACK/NAK) – Attempt #1 dùng chung mẫu nhiễu với phần trên")

attempts = 0
ack = False
ack_trace = []

# Attempt #1
attempts += 1
if ch_type == "AWGN (SNR dB)":
    y_try1 = apply_awgn_with_noise(y_tx, shared_noise)
    rx_sym_try1 = y_try1[samples_per_bit-1::samples_per_bit] if samples_per_bit > 1 else y_try1
    rx_bits_try1 = nrzi_decode(rx_sym_try1, initial_low=initial_low, V_low=V_low, V_high=V_high)
else:
    rx_bits_try1 = apply_bsc_with_flips(tx_bits, shared_flips)

rem_try1 = crc_check_full(rx_bits_try1[:frame_len])
if all(x == 0 for x in rem_try1):
    ack = True
    ack_trace.append(1)
    rx_bits_final = rx_bits_try1
else:
    ack_trace.append(0)

while (not ack) and (attempts < max_retx):
    attempts += 1
    if ch_type == "AWGN (SNR dB)":
        y_try, noise_new, _ = awgn(y_tx, snr_db=snr_db)
        rx_sym_try = y_try[samples_per_bit-1::samples_per_bit] if samples_per_bit > 1 else y_try
        rx_bits_try = nrzi_decode(rx_sym_try, initial_low=initial_low, V_low=V_low, V_high=V_high)
    else:
        rx_bits_try, _ = bsc_flip_bits(tx_bits, p_flip)
    rem_try = crc_check_full(rx_bits_try[:frame_len])
    if all(x == 0 for x in rem_try):
        ack = True
        ack_trace.append(1)
        rx_bits_final = rx_bits_try
        break
    else:
        ack_trace.append(0)

retransmissions = attempts - 1
total_bits = attempts * frame_len
per = 0.0 if ack else 1.0
goodput = (M_len / total_bits) if total_bits > 0 else 0.0

c7, c8, c9, c10 = st.columns(4)
with c7:
    st.metric("Attempts (số lần gửi)", attempts)
with c8:
    st.metric("Retransmissions", retransmissions)
with c9:
    st.metric("Total bits sent", total_bits)
with c10:
    st.metric("PER (sau ARQ)", f"{per:.3f}")
st.metric("Goodput (payload/total bits)", f"{goodput*100:.2f}%")

fig2, ax2 = plt.subplots(1, 1, figsize=(6, 2.6), constrained_layout=True)
plot_ack_trace(ax2, ack_trace if len(ack_trace) > 0 else [0])
st.pyplot(fig2)

if ack:
    text_final = bits_to_text_ascii8(rx_bits_final[:M_len])
    st.success(f"✅ Nhận thành công sau {attempts} lần gửi. ACK phát về TX. Dữ liệu thu: \"{text_final}\"")
else:
    st.error("❌ Hết số lần phát lại cho phép, vẫn lỗi CRC → NAK cuối cùng.")

# ============================================
# Detailed frame info (TX vs RX) + highlight
# ============================================
with st.expander("Chi tiết khung & CRC (TX vs RX + highlight lỗi)"):
    st.write(f"Payload bits (M): {M_len} bit")
    st.code(''.join(str(b) for b in M_bits), language='text')
    st.write(f"FCS (16 bit, TX) = 0x{int(''.join(str(b) for b in fcs_bits), 2):04X}")
    st.code(''.join(str(b) for b in fcs_bits), language='text')
    st.write(f"Frame TX (M||FCS): len={frame_len} bit; HEX = {bits_to_hex(tx_bits)}")

    st.markdown("---")
    st.write("**Frame RX (M||FCS) — đã giải NRZ-I, trước khi kiểm CRC**")
    rx_bits_view = rx_bits[:frame_len]
    st.code(''.join(str(b) for b in rx_bits_view), language='text')

    html = diff_highlight_html(tx_bits, rx_bits_view)
    st.markdown("**RX bits (highlight khác TX):**", unsafe_allow_html=True)
    st.markdown(f"<pre style='white-space:pre-wrap'>{html}</pre>", unsafe_allow_html=True)

    err_idx = [i for i in range(min(len(tx_bits), len(rx_bits_view))) if tx_bits[i] != rx_bits_view[i]]
    st.write(f"Vị trí bit sai (0-based): {err_idx if err_idx else '—'}")

    rem_bits = crc_check_full(rx_bits_view)
    st.write(f"Remainder (RX / G(x)) = {''.join(str(b) for b in rem_bits)} → {'PASS ✅' if all(x==0 for x in rem_bits) else 'FAIL ❌'}")

# ============================================
# Process flow sketch (ASCII diagram for Requirement #1)
# ============================================
with st.expander("Sơ đồ tóm tắt tiến trình xử lý (Requirement #1)"):
    flow = r"""
Text (6 chars)
       ↓  ASCII (8-bit)
  Chuỗi bit M
       ↓  CRC-CCITT (x^16 + x^12 + x^5 + 1)
  Khung TX = M || FCS (16)
       ↓  NRZ-I Encoder (0→đảo, 1→giữ), V_low/V_high
  Dạng sóng mức (TX)
       ↓  Kênh (AWGN dB / BSC p), L ≈ 0 dB
  Dạng sóng mức (RX)
       ↓  Lấy mẫu & Ngưỡng V_th
  Ký hiệu RX (symbol-rate)
       ↓  NRZ-I Decoder (so sánh chuyển mức)
  Chuỗi bit RX
       ↓  CRC check (RX / G(x))
  ACK (PASS) hoặc NAK (FAIL) → (ARQ phát lại)
"""
    st.code(flow, language="text")

st.caption("Ghi chú: NRZ-I không có cơ chế 'phát hiện lỗi ở lớp mã đường' kiểu vi phạm (như AMI/HDB3). Do đó trường hợp (b) trong đề không áp dụng trực tiếp cho NRZ-I; các lỗi được phát hiện nhờ CRC ở lớp kiểm soát lỗi dữ liệu.")
