# ZeroTrace
A Python application that captures network traffic, performs encrypted analysis using homomorphic encryption (TenSEAL), and detects anomalies with machine learning - all while keeping the data encrypted.

## Features:
- **Packet Capture**: Uses Scapy to capture network packets from specified interfaces
- **Homomorphic Encryption**: Leverages TenSEAL (CKKS scheme) for encrypted computations
- **Encrypted ML Analysis**: 
  - Performs anomaly detection on encrypted data using custom linear models
  - Smart feature mapping preserves privacy while enabling ML
- **Statistical Analysis**: 
  - Basic statistical anomaly detection (mean ± n*std)
  - Full performance metrics for all operations
- **Visualization**: Generates histograms of packet length distributions
- **GUI Interface**: Built with customTkinter for modern look and usability
- **Privacy-Preserving**: Data remains encrypted during all computations
