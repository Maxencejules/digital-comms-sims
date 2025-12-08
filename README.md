# Digital Communications Simulation  
A modular Python simulation of a digital communication system featuring **BPSK modulation**, **AWGN channel modeling**, **pulse shaping**, **matched filtering**, **eye diagrams**, **constellation plots**, and **BER analysis**.

This project demonstrates the core DSP concepts used in modern telecom physical layers (optical, wireless, high-speed serial).  
It is designed to be clean, educational, and easily extendable.

---

## 🚀 Features

### ✔ BPSK Modulation & Demodulation
- Binary ↔ symbol mapping (+1 / –1)
- Hard-decision detector

### ✔ AWGN Channel
- Standard \( P_{noise} = P_{signal} / SNR \) implementation  
- Adjustable SNR in dB

### ✔ BER vs SNR Curve (Symbol-Rate Receiver)
- Produces the correct theoretical BER curve for BPSK in AWGN  
- Matches textbook reference performance  
- Script: `main_ber.py`

### ✔ Raised Cosine Pulse Shaping
- Oversampling (SPS)
- Transmit filter (RC)
- Receive matched filter (RC)

### ✔ Eye Diagram
- Visualizes ISI and filter performance  
- Script: `main_eye.py`  
- Produces textbook‑clean eye openings

### ✔ Constellation Diagram
- Scatter plot of sampled BPSK symbols
- Helps visualize noise and impairments  
- Script: `main_constellation.py`

---

## 📁 Project Structure

```
digital-comms-sim/
├── dsp/
│   ├── prbs.py
│   ├── modulation.py
│   ├── channel.py
│   ├── filters.py
│   ├── receiver.py
│   └── __init__.py
│
├── main_ber.py
├── main_eye.py
├── main_constellation.py
├── main_ber_fullchain.py
│
└── README.md
```

---

## 📈 Example Outputs

### **BPSK BER Curve**
![BER Curve](images/ber_curve.png)

### **Eye Diagram**
![Eye Diagram](images/eye_diagram.png)

### **Constellation Diagram**
![Constellation](images/constellation.png)

---

## 🛠 How to Run

### Install dependencies
```
pip install numpy matplotlib
```

### Run demo scripts
```
python main_ber.py
python main_eye.py
python main_constellation.py
```

### Optional (advanced)
```
python main_ber_fullchain.py
```

> Note: `main_ber_fullchain.py` is a work‑in‑progress experiment adding pulse shaping + matched filtering + brute‑force timing recovery.

---

## 🎯 Learning Outcomes

This project demonstrates:

- Digital modulation (BPSK)
- Channel impairments (AWGN)
- Pulse shaping & oversampling
- Matched filtering theory
- Visualization tools used in telecom engineering
- BER analysis & Monte‑Carlo simulation
- Clean Python module design

---

## 📌 Future Work

- QPSK & 16‑QAM
- Root‑raised cosine filters
- Carrier frequency offset simulation
- Clock recovery (Gardner, M&M)
- Soft‑decision demodulation
- Viterbi decoding

---

## 🧑‍💻 Author  
Maxence Jules  
Montreal, QC
