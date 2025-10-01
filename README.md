# NRZI + CRC + Bounded-Noise Demo (Streamlit)

**Live demo:** https://btlcsttnrzicrcabncuniformv6py-xkigj4ek45z5lv4mn224tz.streamlit.app/

A tiny Streamlit app to **visualize a digital link** at a classroom level: NRZ‑I line coding, a **bounded uniform noise** channel, and **CRC‑CCITT** error checking. It’s meant to help students and instructors *see* how data turns into a waveform, how noise may flip bits, and how CRC catches corrupted frames.

---

## Who is this for?
- **Students** taking an introductory *Digital Communications / Information Theory* course.
- **Instructors** who need a quick, interactive demo for **NRZ‑I**, channel noise, and **CRC** checking.

---

## What can it do?
- Convert a 6‑character text to bits, add **CRC‑CCITT**, and **encode with NRZ‑I** (bit **0** flips level, bit **1** holds level).
- Pass the waveform through a **bounded uniform noise** channel (random noise in a fixed range).
- Sample and **decode back**, then show:
  - whether **CRC passes or fails**, 
  - **bit errors (BER)** with positions highlighted,
  - clear **before/after** waveforms and a **3‑panel view** (TX / ±noise band / RX).
- Adjustable parameters: the 6 characters, high/low voltage levels, **SNR (dB)**, and samples per bit.

> Tip: The **SNR slider** includes a “safe/unsafe” cue around **+6 dB** for this setup—below it, flips become likely.

---

## Quick start (local)
Requires **Python 3.11** (or similar). Pinned deps for stability:
```
streamlit==1.31.1
numpy==1.26.4
matplotlib==3.8.4
```
Run:
```bash
pip install -r requirements.txt
python -m streamlit run btl_cstt_nrzi_crc_abnc_uniform_v6.py
```
Open the URL that Streamlit prints in your terminal.

---

## How it works (plain words)
1. **Encode**: Your 6 characters → 48 data bits → add 16 CRC bits → build a transmit frame.  
2. **Line code (NRZ‑I)**: a **0** toggles the level; a **1** keeps it.  
3. **Noise**: we add a random value in a fixed range to each sample (bounded uniform).  
4. **Receive**: sample in the middle of each bit, decide high/low using a mid‑point threshold, then **decode NRZ‑I**.  
5. **Check**: recompute CRC on the received frame. If the remainder is zero → **PASS**, else **FAIL**.
   
> Note about the assignment’s case (b): **NRZ‑I itself does not report “line code violations”** (unlike AMI/HDB3). Error detection in this app is shown by **CRC**.

---

## License
**MIT** — free to use, modify, and share for study/teaching.

---

# (VI) Mô phỏng NRZI + CRC + Kênh nhiễu biên (Streamlit)

**Bản chạy trực tiếp:** https://btlcsttnrzicrcabncuniformv6py-xkigj4ek45z5lv4mn224tz.streamlit.app/

Ứng dụng Streamlit nhỏ gọn để **minh hoạ liên kết số** ở mức *lý thuyết lớp học*: mã đường **NRZ‑I**, kênh **nhiễu đều trong một khoảng cố định**, và **CRC‑CCITT**. Mục tiêu là giúp sinh viên/giảng viên *nhìn thấy* quá trình dữ liệu → dạng sóng, nhiễu làm lật bit, và CRC phát hiện khung lỗi.

---

## Dành cho ai?
- **Sinh viên** học **Cơ sở Thông tin số / Truyền thông số**.
- **Giảng viên** cần **demo tương tác** về **NRZ‑I**, nhiễu kênh và kiểm tra **CRC**.

---

## Ứng dụng làm được gì?
- Chuyển 6 ký tự → bit, gắn **CRC‑CCITT**, **mã hóa NRZ‑I** (bit **0** → đổi mức, bit **1** → giữ mức).
- Truyền qua kênh **nhiễu đều trong đoạn cố định** (bounded uniform).
- Lấy mẫu và **giải mã**, sau đó hiển thị:
  - **CRC PASS/FAIL**, 
  - **BER** kèm highlight vị trí bit sai,
  - **waveform trước/sau** và **3-panel** (TX / dải ±nhiễu / RX).
- Tham số điều chỉnh: 6 ký tự, mức điện áp thấp/cao, **SNR (dB)**, số mẫu/bit.

> Gợi ý: **SNR ~ +6 dB** là ranh giới “an toàn/không an toàn” trong mô phỏng này—dưới mức đó thì lật bit dễ xảy ra.

---

## Chạy nội bộ (local)
Cần **Python 3.11** (khuyến nghị). Thư viện đã ghim phiên bản để ổn định:
```
streamlit==1.31.1
numpy==1.26.4
matplotlib==3.8.4
```
Chạy:
```bash
pip install -r requirements.txt
python -m streamlit run btl_cstt_nrzi_crc_abnc_uniform_v6.py
```
Mở đường dẫn Streamlit in ra trong terminal.

---

## App hoạt động thế nào? (giải thích ngắn)
1. **Mã hoá**: 6 ký tự → 48 bit → cộng 16 bit CRC → khung phát.  
2. **NRZ‑I**: **0** đổi mức; **1** giữ mức.  
3. **Nhiễu**: cộng vào mỗi mẫu một giá trị ngẫu nhiên nằm trong khoảng cố định.  
4. **Thu**: lấy mẫu giữa bit, so ngưỡng trung điểm để quyết định mức, rồi **giải mã NRZ‑I**.  
5. **Kiểm tra**: tính lại CRC. Dư số bằng 0 → **PASS**, khác 0 → **FAIL**.

> Lưu ý về yêu cầu đề (trường hợp b): **NRZ‑I không có “vi phạm mã đường”** để tự báo lỗi như AMI/HDB3. Ví dụ phát hiện lỗi ở đây do **CRC** đảm nhiệm.

---

## Giấy phép
**MIT** — sử dụng/sửa/chia sẻ tự do cho học tập và giảng dạy.
