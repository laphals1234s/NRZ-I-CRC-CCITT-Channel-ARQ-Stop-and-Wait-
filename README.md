# NRZ-I-CRC-CCITT-Channel-ARQ-Stop-and-Wait-

## Overview (EN)

A lightweight Streamlit web app that simulates a simple digital communication system.  
The system transmits a 6-character text message (each encoded as 8-bit Unicode/ASCII), frames it with **CRC-CCITT** for error detection, line-codes with **NRZI** (0 → toggle, 1 → hold), passes through a noisy channel (**AWGN** or **BSC**), then decodes and checks the result.

The app visualizes data, frames, and signals at each step, and highlights bit errors.

## Features (EN)

- **Payload:** exactly 6 characters, 8 bits/char.  
- **Line coding:** NRZI with configurable initial level, V_low, V_high, threshold V_th.  
- **Error detection:** CRC-CCITT (x^16 + x^12 + x^5 + 1, init 0x0000, no final XOR).  
- **Channel models:**  
  - AWGN (controlled by SNR dB)  
  - BSC (bit-flip probability p)  
- **Visualizations & metrics:** TX/RX waveforms, bit/HEX frames, error highlight, SNR/BER, noise stats, CRC PASS/FAIL.  
- **Optional:** Stop-and-Wait ARQ stats (attempts, retransmissions, goodput).

**Note:** NRZI is differential and does not self-detect line violations; error detection is demonstrated via CRC.

## Project Structure

```
.
├── streamlit_nrz_crc_arq_app_v3.py   # Main Streamlit app
├── requirements.txt                  # Dependencies (streamlit, numpy, matplotlib)
└── README.md                         # This file
```

## Run Locally (EN)

**Prerequisites:**  
- Python 3.10+ installed from the official site: https://www.python.org/downloads/

**Steps:**  
1. Download the project (clone or ZIP) and open a terminal in the project folder.  
2. Install dependencies:  
   - Windows (PowerShell):  
     ```
     py -m pip install -r requirements.txt
     ```  
   - macOS / Linux:  
     ```
     python3 -m pip install -r requirements.txt
     ```  
3. Run the app:  
   ```
   streamlit run ./streamlit_nrz_crc_arq_app_v3.py
   ```  
4. Open in browser: Streamlit shows a local URL (usually http://localhost:8501).  
   Press **Ctrl+C** in the terminal to stop the app.  

## Deploy on Streamlit Cloud (EN)

1. Push this repo to GitHub (set it as Public).  
2. Go to https://share.streamlit.io and log in with GitHub.  
3. Click **Create app** → select this repo → choose branch `main` and file `streamlit_nrz_crc_arq_app_v3.py`.  
4. You will get a public link:  
   `https://<your-app-name>.streamlit.app`  

---

## Tổng quan (VI)

Ứng dụng Streamlit mô phỏng một hệ truyền thông số đơn giản.  
Hệ thống truyền đi một đoạn văn bản 6 ký tự (mỗi ký tự mã hoá 8-bit Unicode/ASCII), gắn **CRC-CCITT** để phát hiện lỗi, mã hoá đường bằng **NRZI** (0 → đổi mức, 1 → giữ mức), qua kênh nhiễu (**AWGN** hoặc **BSC**), rồi giải mã và kiểm tra kết quả.

Ứng dụng hiển thị dữ liệu, khung và tín hiệu ở từng bước, đồng thời đánh dấu các bit sai.

## Tính năng (VI)

- **Dữ liệu:** đúng 6 ký tự, 8 bit/ký tự.  
- **Mã đường:** NRZI (cấu hình mức khởi đầu, V_low, V_high, ngưỡng V_th).  
- **Phát hiện lỗi:** CRC-CCITT (đa thức x^16 + x^12 + x^5 + 1, init 0x0000, không XOR cuối).  
- **Mô hình kênh:**  
  - AWGN (điều khiển bằng SNR dB)  
  - BSC (xác suất lật bit p)  
- **Hiển thị & chỉ số:** đồ thị TX/RX, khung nhị phân & HEX, highlight lỗi, SNR/BER, thống kê nhiễu, CRC PASS/FAIL.  
- **Tuỳ chọn:** thống kê ARQ Stop-and-Wait (lần gửi, số lần retransmission, goodput).

**Lưu ý:** NRZI là mã vi phân, không tự phát hiện “vi phạm mã đường”; phát hiện lỗi minh hoạ bằng CRC.

## Cấu trúc dự án

```
.
├── streamlit_nrz_crc_arq_app_v3.py   # Ứng dụng Streamlit chính
├── requirements.txt                  # Thư viện (streamlit, numpy, matplotlib)
└── README.md                         # Tệp hướng dẫn này
```

## Chạy cục bộ (VI)

**Yêu cầu:**  
- Python 3.10+ cài từ trang chủ: https://www.python.org/downloads/

**Các bước:**  
1. Tải dự án (clone hoặc ZIP) và mở terminal tại thư mục dự án.  
2. Cài thư viện:  
   - Windows (PowerShell):  
     ```
     py -m pip install -r requirements.txt
     ```  
   - macOS / Linux:  
     ```
     python3 -m pip install -r requirements.txt
     ```  
3. Chạy ứng dụng:  
   ```
   streamlit run ./streamlit_nrz_crc_arq_app_v3.py
   ```  
4. Mở trình duyệt theo URL (thường là http://localhost:8501).  
   Nhấn **Ctrl+C** để dừng ứng dụng.  

## Triển khai trên Streamlit Cloud (VI)

1. Đẩy repo này lên GitHub (đặt Public).  
2. Vào https://share.streamlit.io, đăng nhập bằng GitHub.  
3. Chọn **Create app** → repo này → branch `main` và file `streamlit_nrz_crc_arq_app_v3.py`.  
4. Bạn sẽ nhận được link công khai:  
   `https://<your-app-name>.streamlit.app`  

---

## License

This project is licensed under the **MIT License** – free to use, modify, and distribute.  
(See the [LICENSE](./LICENSE) file for full text.)
