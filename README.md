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

## Output :
![Image](https://github.com/user-attachments/assets/dfc69e7e-c174-4e2e-9bde-3bf8bb566e6a)

![Image](https://github.com/user-attachments/assets/5b06488f-7173-4f27-ad94-00bf851b8421)

![Image](https://github.com/user-attachments/assets/5959515c-2664-477d-8864-2009af9616af)

## Technology Stack

- Python 3.8+
- Libraries:
  - TenSEAL (Microsoft SEAL wrapper) - Homomorphic encryption
  - Scapy - Network packet capture
  - customTkinter - Modern GUI
  - NumPy, Matplotlib - Data analysis & visualization
  - scikit-learn - Machine learning models (Isolation Forest)




   
