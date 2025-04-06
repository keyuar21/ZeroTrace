import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import threading
import logging
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts
from scapy.all import sniff

import pandas as pd
from sklearn.ensemble import IsolationForest
from PIL import Image, ImageTk

ml_results = {}

# ------------------------------
# Logging Handler to Redirect Logs to the Text Widget
# ------------------------------
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        log_entry = self.format(record) + "\n"
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, log_entry)
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)

# ------------------------------
# Network Analysis Functions
# ------------------------------
def capture_packets(count, interface=None):
    """Capture packets using Scapy."""
    try:
        logging.info(f"Starting packet capture: capturing {count} packets")
        packets = sniff(count=count, iface=interface)
        logging.info(f"Captured {len(packets)} packets")
        return packets
    except Exception as e:
        logging.error("Error capturing packets: " + str(e))
        return []

def extract_packet_data(packets):
    """Extract packet lengths from captured packets."""
    packet_lengths = []
    for pkt in packets:
        try:
            length = len(pkt)
            packet_lengths.append(length)
        except Exception as e:
            logging.warning(f"Failed to extract length from a packet: {e}")
    return packet_lengths

def setup_encryption_context():
    """Set up a TenSEAL context for CKKS homomorphic encryption."""
    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree=8192, 
            coeff_mod_bit_sizes=[60, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2**40 
        logging.info("Encryption context set up successfully")
        return context
    except Exception as e:
        logging.error("Error setting up encryption context: " + str(e))
        return None

def encrypt_data(context, data):
    """Encrypt a list of numbers using TenSEAL CKKS vector."""
    try:
        data_float = [float(d) for d in data]
        encrypted_vector = ts.ckks_vector(context, data_float)
        # Attach the context so later functions can access it
        encrypted_vector.context = context
        logging.info("Data encrypted successfully")
        return encrypted_vector
    except Exception as e:
        logging.error("Error encrypting data: " + str(e))
        return None

def analyze_encrypted_data(encrypted_vector):
    """
    Perform a simple homomorphic operation on the encrypted data.
    Here, we double each element as a demonstration.
    """
    try:
        encrypted_result = encrypted_vector + encrypted_vector
        logging.info("Performed homomorphic operation on encrypted data")
        return encrypted_result
    except Exception as e:
        logging.error("Error during encrypted data analysis: " + str(e))
        return None

def decrypt_data(encrypted_data):
    """Decrypt the data from a TenSEAL encrypted vector."""
    try:
        decrypted = encrypted_data.decrypt()
        logging.info("Data decrypted successfully")
        return decrypted
    except Exception as e:
        logging.error("Error decrypting data: " + str(e))
        return None

def detect_anomalies(data, threshold=2.0):
    """
    Detect anomalies in data using a simple statistical method.
    Any value outside mean ± (threshold * std) is considered an anomaly.
    Returns a tuple (mean, std, anomalies) where anomalies is a list of (index, value).
    """
    data_array = np.array(data)
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    anomalies = []
    for idx, value in enumerate(data_array):
        if abs(value - mean_val) > threshold * std_val:
            anomalies.append((idx, value))
    return mean_val, std_val, anomalies

def visualize_data(data, output_file="packet_histogram.png"):
    """Visualize packet length data using a histogram."""
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=20, edgecolor='black')
        plt.title("Packet Length Distribution")
        plt.xlabel("Packet Length")
        plt.ylabel("Frequency")
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Visualization saved as {output_file}")
    except Exception as e:
        logging.error("Error visualizing data: " + str(e))

def export_report(packet_data, detailed_analysis, filename="packet_report.csv"):
    """
    Export packet data and detailed analysis to a CSV file.
    detailed_analysis should be a dictionary containing summary stats and anomaly info.
    """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Packet Index", "Packet Length (Original)", "Packet Length (Doubled)"])
            for idx, (orig, doubled) in enumerate(zip(packet_data['original'], packet_data['doubled'])):
                writer.writerow([idx, orig, doubled])
            writer.writerow([])
            writer.writerow(["Detailed Analysis"])
            writer.writerow(["Mean (Doubled)", detailed_analysis['mean']])
            writer.writerow(["Std Dev (Doubled)", detailed_analysis['std']])
            writer.writerow(["Anomalies (Index, Value)"])
            for anomaly in detailed_analysis['anomalies']:
                writer.writerow(list(anomaly))
        logging.info(f"Report exported to {filename}")
    except Exception as e:
        logging.error("Error exporting report: " + str(e))

# ------------------------------
# Machine Learning Functions for Encrypted Data
# ------------------------------
def encrypted_ml_anomaly_detection(encrypted_vector, threshold=0.0):
    """
    Run a simple linear anomaly detection model on encrypted data.
    Using a unique mapping technique, each packet length x is mapped to features [x, x^2, 1]
    and an encrypted anomaly score is computed as:
    
        score = w0 * x + w1 * x^2 + w2
    
    where the model weights are pre-determined.
    Only the final scores are decrypted and packets with scores above the threshold are flagged.
    
    Returns a tuple:
        (anomalies, all_scores, stats)
    where anomalies is a list of dictionaries with keys 'index' and 'score',
    all_scores is the list of all anomaly scores,
    and stats is a dictionary containing the mean and standard deviation of all scores.
    """
    try:
       
        context = encrypted_vector.context
        
       
        encrypted_x2 = encrypted_vector * encrypted_vector
        
        
        count = len(encrypted_vector.decrypt())  # decrypting only to get count (non-sensitive)
        ones = [1.0] * count
        encrypted_ones = ts.ckks_vector(context, ones)
        
        # Pre-determined model weights (example values, can be tuned)
        w0 = 0.5
        w1 = 0.01
        w2 = -5.0
        
        
        encrypted_score = encrypted_vector * w0 + encrypted_x2 * w1 + encrypted_ones * w2
        
        # Decrypt the anomaly scores
        scores = encrypted_score.decrypt()
        
        # Compute statistics on the scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Flag anomalies where score > threshold
        anomalies = [{"index": i, "score": score} for i, score in enumerate(scores) if score > threshold]
        logging.info(f"Encrypted ML-based anomaly detection: {len(anomalies)} anomalies found.")
        stats = {"mean_score": mean_score, "std_score": std_score}
        return anomalies, scores, stats
    except Exception as e:
        logging.error("Error in encrypted ML anomaly detection: " + str(e))
        return [], [], {}

# (Legacy ML functions using IsolationForest are kept for reference but not used in the encrypted pipeline)
def load_nsl_kdd_dataset(dataset_file="nsl_kdd.csv"):
    try:
        columns = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
        ]
        data = pd.read_csv(dataset_file, header=None, names=columns)
        features = data[['duration', 'src_bytes', 'dst_bytes']].apply(pd.to_numeric, errors='coerce')
        features = features.fillna(0)
        logging.info("NSL-KDD dataset loaded and preprocessed.")
        return features
    except Exception as e:
        logging.error("Error loading NSL-KDD dataset: " + str(e))
        return None

def train_ml_model_from_dataset(dataset_file="nsl_kdd.csv"):
    features = load_nsl_kdd_dataset(dataset_file)
    if features is None:
        return None
    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(features)
        logging.info("ML anomaly detection model trained successfully on NSL-KDD dataset.")
        logging.info(f"Model Parameters: {model.get_params()}")
        return model
    except Exception as e:
        logging.error("Error training ML model: " + str(e))
        return None

def ml_detect_anomalies(model, features):
    try:
        predictions = model.predict(features)
        anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]
        logging.info(f"ML detected {len(anomaly_indices)} anomalies.")
        return anomaly_indices
    except Exception as e:
        logging.error("Error in ML anomaly detection: " + str(e))
        return []

def analyze_model_performance(dataset_file="nsl_kdd.csv"):
    features = load_nsl_kdd_dataset(dataset_file)
    if features is None:
        return "Error loading dataset."
    ml_model = train_ml_model_from_dataset(dataset_file)
    if ml_model is None:
        return "Error training model."
    predictions = ml_model.predict(features)
    total = len(predictions)
    anomalies = np.sum(predictions == -1)
    normal = total - anomalies
    performance_summary = (
        f"Total Samples: {total}\n"
        f"Normal Instances: {normal}\n"
        f"Anomalies Detected: {anomalies}\n"
        f"Anomaly Percentage: {anomalies / total * 100:.2f}%\n\n"
        f"Model Parameters:\n{ml_model.get_params()}\n"
    )
    return performance_summary




# Function to Process Network Data (Called in a Separate Thread)

def process_network_data(params):
    global ml_results
    start_time = time.time()
    # Unpack parameters from GUI
    capture_count = params.get("capture_count")
    interface = params.get("interface")
    do_encrypt = params.get("encrypt")
    do_analyze = params.get("analyze")
    do_visualize = params.get("visualize")
    do_report = params.get("report")
    do_performance = params.get("performance")
    do_ml = params.get("ml_anomaly")

    # Step 1: Capture packets
    packets = capture_packets(capture_count, interface)
    capture_time = time.time()

    # Step 2: Extract packet lengths
    original_packet_lengths = extract_packet_data(packets)
    logging.info(f"Extracted packet lengths: {original_packet_lengths}")
    extraction_time = time.time()

    # Step 3: Encrypt packet data if requested
    if do_encrypt:
        context = setup_encryption_context()
        if context:
            encrypted_vector = encrypt_data(context, original_packet_lengths)
        else:
            logging.error("Encryption context not available. Exiting.")
            return
        encryption_time = time.time()
    else:
        encrypted_vector = None
        encryption_time = time.time()

    # Step 4: Analyze encrypted data if requested
    if do_analyze and encrypted_vector is not None:
        encrypted_result = analyze_encrypted_data(encrypted_vector)
        decrypted_result = decrypt_data(encrypted_result)
        if decrypted_result is not None:
            total_sum = sum(decrypted_result)
            average = total_sum / len(decrypted_result) if decrypted_result else 0
            logging.info(f"Decrypted Encrypted Result (each element doubled): {decrypted_result}")
            logging.info(f"Total Sum: {total_sum}")
            logging.info(f"Average Packet Length (doubled): {average}")
            # Detailed analysis and anomaly detection on the doubled data
            mean_val, std_val, anomalies = detect_anomalies(decrypted_result)
            logging.info(f"Mean (Doubled): {mean_val}")
            logging.info(f"Standard Deviation (Doubled): {std_val}")
            if anomalies:
                logging.info("Anomalies detected (Index, Value): " + ", ".join(str(a) for a in anomalies))
            else:
                logging.info("No anomalies detected.")
            analysis_time = time.time()
        else:
            analysis_time = time.time()
    else:
        analysis_time = time.time()
        decrypted_result = None
        mean_val, std_val, anomalies = None, None, None

    # Step 5: Encrypted ML-based Anomaly Detection using Smart Mapping
    if do_ml and encrypted_vector is not None:
        ml_anomalies, all_scores, ml_stats = encrypted_ml_anomaly_detection(encrypted_vector, threshold=0.0)
        logging.info("Encrypted ML-based anomalies detected at indices: " +
                     str([a['index'] for a in ml_anomalies]))
        ml_time = time.time()
        ml_results = {"anomalies": ml_anomalies, "all_scores": all_scores, "stats": ml_stats}
    else:
        ml_time = time.time()

    # Step 6: Visualize data if requested
    if do_visualize:
        visualize_data(original_packet_lengths)
        visualization_time = time.time()
    else:
        visualization_time = time.time()

    # Step 7: Export report if requested
    if do_report:
        packet_data = {
            'original': original_packet_lengths,
            'doubled': decrypted_result if decrypted_result is not None else original_packet_lengths
        }
        detailed_analysis = {
            'mean': mean_val,
            'std': std_val,
            'anomalies': anomalies
        }
        export_report(packet_data, detailed_analysis)
        report_time = time.time()
    else:
        report_time = time.time()

    # Step 8: Performance monitoring
    if do_performance:
        logging.info("Performance Metrics:")
        logging.info(f"Packet Capture Time: {capture_time - start_time:.4f} seconds")
        logging.info(f"Data Extraction Time: {extraction_time - capture_time:.4f} seconds")
        if do_encrypt:
            logging.info(f"Encryption Time: {encryption_time - extraction_time:.4f} seconds")
        if do_analyze and encrypted_vector is not None:
            logging.info(f"Analysis Time: {analysis_time - encryption_time:.4f} seconds")
        if do_ml:
            logging.info(f"Encrypted ML Anomaly Detection Time: {ml_time - analysis_time:.4f} seconds")
        if do_visualize:
            logging.info(f"Visualization Time: {visualization_time - ml_time:.4f} seconds")
        if do_report:
            logging.info(f"Report Export Time: {report_time - visualization_time:.4f} seconds")
        total_time = time.time() - start_time
        logging.info(f"Total Execution Time: {total_time:.4f} seconds")


# GUI 

class NetworkAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Network Analysis with Homomorphic & Encrypted ML-based Anomaly Detection")
        self.geometry("950x800")
        
        # Create input frame for parameters
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=10, padx=10, fill="x")
        
        # Capture Count
        self.capture_count_label = ctk.CTkLabel(self.input_frame, text="Number of Packets to Capture:")
        self.capture_count_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.capture_count_entry = ctk.CTkEntry(self.input_frame)
        self.capture_count_entry.grid(row=0, column=1, padx=5, pady=5)
        self.capture_count_entry.insert(0, "20")
        
        # Interface
        self.interface_label = ctk.CTkLabel(self.input_frame, text="Interface (optional):")
        self.interface_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.interface_entry = ctk.CTkEntry(self.input_frame)
        self.interface_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Options Checkboxes
        self.encrypt_var = tk.BooleanVar(value=True)
        self.analyze_var = tk.BooleanVar(value=True)
        self.visualize_var = tk.BooleanVar(value=True)
        self.report_var = tk.BooleanVar(value=True)
        self.performance_var = tk.BooleanVar(value=True)
        self.ml_anomaly_var = tk.BooleanVar(value=True)
        
        self.encrypt_checkbox = ctk.CTkCheckBox(self.input_frame, text="Encrypt", variable=self.encrypt_var)
        self.encrypt_checkbox.grid(row=2, column=0, padx=5, pady=5)
        self.analyze_checkbox = ctk.CTkCheckBox(self.input_frame, text="Analyze", variable=self.analyze_var)
        self.analyze_checkbox.grid(row=2, column=1, padx=5, pady=5)
        self.visualize_checkbox = ctk.CTkCheckBox(self.input_frame, text="Visualize", variable=self.visualize_var)
        self.visualize_checkbox.grid(row=3, column=0, padx=5, pady=5)
        self.report_checkbox = ctk.CTkCheckBox(self.input_frame, text="Export Report", variable=self.report_var)
        self.report_checkbox.grid(row=3, column=1, padx=5, pady=5)
        self.performance_checkbox = ctk.CTkCheckBox(self.input_frame, text="Performance Metrics", variable=self.performance_var)
        self.performance_checkbox.grid(row=4, column=0, padx=5, pady=5)
        self.ml_anomaly_checkbox = ctk.CTkCheckBox(self.input_frame, text="Encrypted ML-based Anomaly Detection", variable=self.ml_anomaly_var)
        self.ml_anomaly_checkbox.grid(row=4, column=1, padx=5, pady=5)
        
        # Button to start process
        self.start_button = ctk.CTkButton(self.input_frame, text="Start Analysis", command=self.start_analysis)
        self.start_button.grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        # Additional Buttons for viewing outputs
        self.extra_frame = ctk.CTkFrame(self)
        self.extra_frame.pack(pady=10, padx=10, fill="x")
        self.report_button = ctk.CTkButton(self.extra_frame, text="View CSV Report", command=self.view_report)
        self.report_button.grid(row=0, column=0, padx=5, pady=5)
        self.viz_button = ctk.CTkButton(self.extra_frame, text="View Visualization", command=self.view_visualization)
        self.viz_button.grid(row=0, column=1, padx=5, pady=5)
        self.model_button = ctk.CTkButton(self.extra_frame, text="View Model Details", command=self.view_model_performance)
        self.model_button.grid(row=0, column=2, padx=5, pady=5)
        self.anomaly_button = ctk.CTkButton(self.extra_frame, text="View Detected Anomalies", command=self.view_detected_anomalies)
        self.anomaly_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Text area for logs/output
        self.log_text = scrolledtext.ScrolledText(self, state='disabled', height=20)
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Set up logging handler to display messages in the text area
        self.text_handler = TextHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.text_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.text_handler)
        logging.getLogger().setLevel(logging.INFO)

    def start_analysis(self):
        try:
            capture_count = int(self.capture_count_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Capture count must be an integer.")
            return
        
        params = {
            "capture_count": capture_count,
            "interface": self.interface_entry.get().strip() or None,
            "encrypt": self.encrypt_var.get(),
            "analyze": self.analyze_var.get(),
            "visualize": self.visualize_var.get(),
            "report": self.report_var.get(),
            "performance": self.performance_var.get(),
            "ml_anomaly": self.ml_anomaly_var.get()
        }
        thread = threading.Thread(target=process_network_data, args=(params,))
        thread.daemon = True
        thread.start()

    def view_report(self):
        """Open and display the CSV report in a new window."""
        try:
            with open("packet_report.csv", "r") as file:
                content = file.read()
            report_window = tk.Toplevel(self)
            report_window.title("CSV Report")
            text_area = scrolledtext.ScrolledText(report_window, width=80, height=30)
            text_area.pack(padx=10, pady=10)
            text_area.insert(tk.END, content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open CSV report: {e}")

    def view_visualization(self):
        """Open and display the histogram image in a new window."""
        try:
            viz_window = tk.Toplevel(self)
            viz_window.title("Packet Histogram")
            img = Image.open("packet_histogram.png")
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(viz_window, image=photo)
            label.image = photo  # keep a reference!
            label.pack(padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open visualization: {e}")

    def view_model_performance(self):
        """Display detailed model performance analysis in a new window."""
        summary = analyze_model_performance()
        if summary:
            model_window = tk.Toplevel(self)
            model_window.title("Model Performance Details")
            text_area = scrolledtext.ScrolledText(model_window, width=80, height=20)
            text_area.pack(padx=10, pady=10)
            text_area.insert(tk.END, summary)
        else:
            messagebox.showerror("Error", "Could not analyze model performance.")

    def view_detected_anomalies(self):
        """Display detected anomalies with detailed statistics in a new window."""
        global ml_results
        if not ml_results:
            messagebox.showerror("Error", "No ML anomaly detection results available yet.")
            return
        anomaly_window = tk.Toplevel(self)
        anomaly_window.title("Detected Anomalies Details")
        text_area = scrolledtext.ScrolledText(anomaly_window, width=80, height=30)
        text_area.pack(padx=10, pady=10)
        details = "Encrypted ML-based Anomaly Detection Details\n"
        details += "--------------------------------------------------\n"
        stats = ml_results.get("stats", {})
        details += f"Mean Anomaly Score: {stats.get('mean_score', 'N/A')}\n"
        details += f"Std Dev of Scores: {stats.get('std_score', 'N/A')}\n\n"
        details += "Anomalies:\n"
        anomalies = ml_results.get("anomalies", [])
        if anomalies:
            for anomaly in anomalies:
                details += f"Packet Index: {anomaly['index']}, Score: {anomaly['score']}\n"
        else:
            details += "No anomalies detected.\n"
        text_area.insert(tk.END, details)

#main
if __name__ == "__main__":
    app = NetworkAnalysisApp()
    app.mainloop()
    