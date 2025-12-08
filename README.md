# 📡 Digital Communications Simulation

A modular Python simulation of a digital communication system featuring:

- **BPSK modulation & demodulation**
- **AWGN channel modeling**
- **Raised cosine pulse shaping**
- **Matched filtering**
- **Eye diagrams**
- **Constellation plots**
- **BER vs SNR analysis**
- **Full transmitter → channel → receiver chain simulation**

This project demonstrates the core DSP techniques used in modern telecom PHY systems  
(wireless, optical, and high-speed digital links).

---

# 🚀 Features

## ✔ BPSK Modulation & Demodulation
- Symbol mapping: {0 → +1, 1 → –1}  
- Hard-decision slicing at the receiver  
- Clean modular implementation (`dsp/modulation.py`, `dsp/receiver.py`)

---

## ✔ AWGN Channel
Standard AWGN model:

\[
y = x + n,\quad n \sim \mathcal{N}(0, \sigma^2)
\]

with noise variance computed from SNR(dB).

File: `dsp/channel.py`

---

## ✔ BER vs SNR (Symbol-Rate Receiver)
A classic BPSK-in-AWGN BER experiment.

- Matches theoretical curve shape  
- Validates modulation, noise model, and demodulation flow  
- Script: `main_ber.py`

Plot (generated after running the script):

![BER Curve](images/ber_curve.png)

---

## ✔ Eye Diagram
Shows intersymbol interference (ISI) and raised-cosine pulse shaping behavior.

Script: `main_eye.py`

![Eye Diagram](images/eye_diagram.png)

---

## ✔ Constellation Diagram
Scatter plot of demodulated BPSK samples.

Script: `main_constellation.py`

![Constellation](images/constellation.png)

---

# 🌐 Full Pulse-Shaped BPSK BER (Transmitter → Channel → Receiver)

This test measures BER for the **complete** digital communications chain:

### Transmitter
- PRBS bit source  
- BPSK mapper  
- Oversampling  
- Raised cosine pulse shaping  

### Channel
- AWGN injection at user-selected SNR(dB)

### Receiver
- Matched filtering (time-reversed RC)  
- **Timing phase search (0…SPS-1) for robust symbol sampling**  
- Hard-decision BPSK demodulation  

Script: `main_ber_fullchain.py`

---

## 🧪 Why this experiment matters

This test demonstrates *real-world* impairments:

- ISI introduced by pulse shaping  
- Noise **after** filtering  
- Sampling phase misalignment  
- True end-to-end BER measurement  

This is far more realistic than the symbol-rate BER test.

---

## 📉 Example Output (after timing fix)

```
SNR =  0 dB -> BER ≈ 2.8e-03
SNR =  2 dB -> BER ≈ 1.6e-04
SNR =  4 dB -> BER = 0
SNR =  6+ dB -> BER = 0
```

This confirms correct pulse shaping, matched filtering, and detection alignment.

![Full-Chain BER](/images/fullchain_ber.png)


---

# 📁 Project Structure

```markdown
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
├── images/              # Saved plots go here
└── README.md
```

---

# 🛠 Running the Simulations

## Install dependencies

```bash
pip install numpy matplotlib
```

## Run individual experiments

```bash
python main_ber.py
python main_eye.py
python main_constellation.py
python main_ber_fullchain.py
```

---

# 🎯 Learning Outcomes

Working with this project helps build understanding of:

- Digital modulation (BPSK)
- Oversampling and raised-cosine pulse shaping
- Matched filtering theory
- Noise modeling
- Eye and constellation visualization
- BER Monte-Carlo estimation
- Timing recovery concepts
- Clean modular DSP code design

---

# 🚧 Future Extensions

- QPSK, 16-QAM modulation
- Root-raised cosine (RRC) filters
- Carrier frequency offset (CFO)
- Gardner or Mueller & Müller timing recovery
- Soft-decision demodulation
- Viterbi or LDPC decoding

---

# 🧑‍💻 Author

**Maxence Jules**  
Montreal, QC
