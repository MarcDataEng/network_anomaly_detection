# Network Flow Analysis and Anomaly Detection Script
# This script processes PCAP files to extract, analyze and detect network flow anomalies
# It implements feature engineering for network security analysis

import sys
import os  # For interacting with the file system
import pandas as pd  # For handling dataframes and CSVs
import numpy as np
from datetime import datetime
from tqdm import tqdm  # Import tqdm for progress tracking
from scapy import all
from nfstream import NFStreamer  # For working with PCAP files and flow analysis
import csv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

print("Starting network flow analysis and anomaly detection process...")

# Specify the directory containing the .pcap files
pcap_directory = 'pcap_files'
print(f"Looking for PCAP files in directory: {pcap_directory}")

# Initialize an empty list to store all flows data
all_flows_data = []

# Iterate through each .pcap file in the directory
for file in os.listdir(pcap_directory):
    # Check if the file is a .pcap file
    if file.endswith('.pcap'):
        full_path = os.path.join(pcap_directory, file)  # Get the full path of the file
        print(f"Processing file: {full_path}")

        # Create an NFStreamer instance with statistical analysis enabled
        # Set timeouts to handle both short and long-lived flows
        my_streamer = NFStreamer(
            source=full_path,
            statistical_analysis=True,
            idle_timeout=60,  # 60 seconds idle timeout
            active_timeout=120  # 120 seconds active timeout
        )

        # List to store extracted flow data for this file
        file_flows_data = []
        total_flows = len(list(my_streamer))  # Count total flows for tqdm
        my_streamer = NFStreamer(source=full_path, statistical_analysis=True,
                                 idle_timeout=60, active_timeout=120)  # Re-create the streamer

        print(f"Found {total_flows} flows in {file}")

        # Using tqdm to track progress for flow extraction
        with tqdm(total=total_flows, desc=f"Extracting flows from {file}", unit="flows") as pbar:
            for flow in my_streamer:
                # Extract comprehensive flow metrics including statistical features
                flow_data = {
                    'src_ip': flow.src_ip,
                    'dst_ip': flow.dst_ip,
                    'src_port': flow.src_port,
                    'dst_port': flow.dst_port,
                    'protocol': flow.protocol,
                    'application_name': flow.application_name,
                    'bidirectional_packets': flow.bidirectional_packets,
                    'bidirectional_bytes': flow.bidirectional_bytes,
                    'bidirectional_first_seen_ms': flow.bidirectional_first_seen_ms,
                    'bidirectional_last_seen_ms': flow.bidirectional_last_seen_ms,

                    # Statistical features for anomaly detection
                    'bidirectional_mean_ps': flow.bidirectional_mean_ps,  # Packet size statistics
                    'bidirectional_stddev_ps': flow.bidirectional_stddev_ps,
                    'src2dst_mean_ps': flow.src2dst_mean_ps,
                    'src2dst_stddev_ps': flow.src2dst_stddev_ps,
                    'dst2src_mean_ps': flow.dst2src_mean_ps,
                    'dst2src_stddev_ps': flow.dst2src_stddev_ps,

                    # Packet Inter-Arrival Time (PIAT) statistics
                    'bidirectional_mean_piat_ms': flow.bidirectional_mean_piat_ms,
                    'bidirectional_stddev_piat_ms': flow.bidirectional_stddev_piat_ms,
                    'src2dst_mean_piat_ms': flow.src2dst_mean_piat_ms,
                    'src2dst_stddev_piat_ms': flow.src2dst_stddev_piat_ms,
                    'dst2src_mean_piat_ms': flow.dst2src_mean_piat_ms,
                    'dst2src_stddev_piat_ms': flow.dst2src_stddev_piat_ms
                }
                file_flows_data.append(flow_data)
                pbar.update(1)  # Update progress bar

        # Append the current file's data to the all_flows_data list
        all_flows_data.extend(file_flows_data)

print("Flow extraction completed. Creating DataFrame...")

# Convert the list of flow data into a pandas DataFrame
flows_df = pd.DataFrame(all_flows_data)

# Display the DataFrame's summary
print("DataFrame created with the following structure:")
print(flows_df.info())

# Save raw flows to CSV
output_path = "csv_files/raw_flows.csv"
flows_df.to_csv(output_path, index=False)
print(f"Raw flows DataFrame saved to {output_path}")

# Define the time window T in milliseconds (used for fan-in/fan-out calculation)
T = 10  # 10 seconds window
print(f"\nCalculating network features using {T} second time window...")

def calculate_features(df, T):
    """
    Calculate fan-in and fan-out metrics for each flow within a sliding time window
    
    Parameters:
    df (DataFrame): Input flows DataFrame
    T (int): Time window in seconds
    
    Returns:
    DataFrame: Enriched DataFrame with fan-in/fan-out metrics
    """
    # Convert period T to milliseconds
    T_ms = T * 1000

    # Initialize new columns for fan metrics
    df['fan_out_src'] = 0  # Number of unique destinations from source
    df['fan_in_dst'] = 0   # Number of unique sources to destination
    df['fan_in_src'] = 0   # Number of unique sources to source
    df['fan_out_dst'] = 0  # Number of unique destinations from destination

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating fan metrics", unit="flows"):
        # Define sliding time window
        mid_timestamp = row['bidirectional_first_seen_ms']
        time_window_start = mid_timestamp - (T_ms // 2)
        time_window_end = mid_timestamp + (T_ms // 2)

        # Filter flows within time window
        window_df = df[(df['bidirectional_first_seen_ms'] >= time_window_start) & 
                      (df['bidirectional_first_seen_ms'] <= time_window_end)].copy()

        # Calculate fan metrics
        df.at[i, 'fan_in_src'] = window_df[window_df['dst_ip'] == row['src_ip']]['src_ip'].nunique()
        df.at[i, 'fan_out_src'] = window_df[window_df['src_ip'] == row['src_ip']]['dst_ip'].nunique()
        df.at[i, 'fan_in_dst'] = window_df[window_df['dst_ip'] == row['dst_ip']]['src_ip'].nunique()
        df.at[i, 'fan_out_dst'] = window_df[window_df['src_ip'] == row['dst_ip']]['dst_ip'].nunique()

    return df

# Apply the feature calculation function
print("Calculating fan-in/fan-out metrics...")
df = calculate_features(flows_df, T)

# Save the enriched dataframe to a CSV file
csv_filename = 'csv_files/enriched_flows.csv'
df.to_csv(csv_filename, index=False)
print(f"Enriched flows saved to {csv_filename}")

print("\nLoading ground truth data for flow labeling...")
# Load ground truth file for labeling
gt_file = 'pcap_files/flows/TRAIN.gt'
train_gt_df = pd.read_csv(gt_file)

# Convert necessary columns to compatible types
df['src_port'] = df['src_port'].astype(int)
df['dst_port'] = df['dst_port'].astype(int)

def match_label(row):
    """
    Match flow with ground truth data to determine if it's anomalous
    Returns 1 for anomalous flows, 0 for normal flows
    """
    # Match based on ports, protocol, and timestamp overlap
    matched = train_gt_df[
        (train_gt_df['src_port'] == row['src_port']) &
        (train_gt_df['dst_port'] == row['dst_port']) &
        (train_gt_df['protocol'] == row['protocol']) &
        (train_gt_df['first_timestamp_ms'] <= row['bidirectional_last_seen_ms']) &
        (train_gt_df['last_timestamp_ms'] >= row['bidirectional_first_seen_ms'])
    ]
    return 1 if not matched.empty else 0

print("Labeling flows based on ground truth data...")
# Apply labeling function
df['label'] = df.apply(match_label, axis=1)

# Save labeled DataFrame
df.to_csv('csv_files/labeled_flows.csv', index=False)
print(f"Labeled flows saved to {output_path}")

def get_first_octet(ip):
    """Extract first octet from IP address"""
    return int(ip.split('.')[0])

def get_ip_class(ip):
    """
    Determine IP address class based on first octet
    Returns: Class A, B, C, D (multicast), or E (reserved)
    """
    first_octet = get_first_octet(ip)
    
    if 0 <= first_octet <= 127:  # Class A
        return 'Class A'
    elif 128 <= first_octet <= 191:  # Class B
        return 'Class B'
    elif 192 <= first_octet <= 223:  # Class C
        return 'Class C'
    elif 224 <= first_octet <= 239:  # Class D (multicast)
        return 'Class D (multicast)'
    elif 240 <= first_octet <= 255:  # Class E (reserved)
        return 'Class E (reserved)'
    else:
        return 'Unknown'

print("\nEnriching data with IP classification...")
# Apply IP classification
df['src_ip_class'] = df['src_ip'].apply(get_ip_class)
df['dst_ip_class'] = df['dst_ip'].apply(get_ip_class)
df = pd.get_dummies(df, columns=['src_ip_class', 'dst_ip_class'])

# Calculate flow duration
df['bidirectional_duration_ms'] = df['bidirectional_last_seen_ms'] - df['bidirectional_first_seen_ms']

print("Standardizing numerical features...")
# Select numerical columns for standardization
numeric_cols = ['bidirectional_bytes', 'bidirectional_duration_ms',
       'bidirectional_mean_piat_ms', 'bidirectional_mean_ps',
       'bidirectional_packets', 'bidirectional_stddev_piat_ms',
       'bidirectional_stddev_ps', 'dst2src_mean_piat_ms', 'dst2src_mean_ps',
       'dst2src_stddev_piat_ms', 'dst2src_stddev_ps', 'fan_in_dst',
       'fan_in_src', 'fan_out_dst', 'fan_out_src', 'src2dst_mean_piat_ms',
       'src2dst_mean_ps', 'src2dst_stddev_piat_ms', 'src2dst_stddev_ps']

# Apply StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Processing protocol information...")
# Handle protocol encoding
most_frequent_values = [6, 17, 89, 1, 132]  # Most common protocol numbers
df['protocol'] = df['protocol'].where(df['protocol'].isin(most_frequent_values), 0)
df = pd.get_dummies(df, columns=['protocol'])

def get_port_cat(port):
    """
    Categorize ports based on IANA assignments:
    - Well-known ports: 0-1023
    - Registered ports: 1024-49151
    - Dynamic ports: 49152-65535
    """    
    if 0 <= port <= 1023:
        return 'WellKnown'
    elif 1024 <= port <= 49151:
        return 'Registered'
    elif 49152 <= port <= 65535:
        return 'Dynamic'
    else:
        return 'Unknown'

print("Categorizing ports...")
# Apply port categorization
df['src_port_class'] = df['src_port'].apply(get_port_cat)
df['dst_port_class'] = df['dst_port'].apply(get_port_cat)
df = pd.get_dummies(df, columns=['src_port_class', 'dst_port_class'])

# Save encoded DataFrame
csv_filename = 'csv_files/encoded_flows.csv'
df.to_csv(csv_filename, index=False)
print(f'Encoded DataFrame saved to {csv_filename}')

def create_connection_pattern_features(df):
    """
    Create features based on connection patterns using fan-in/fan-out metrics
    and IP class relationships
    """
    print("Creating connection pattern features...")
    features = df.copy()
    
    # Fan-in/fan-out ratios
    features['fan_ratio_src'] = features['fan_out_src'] / (features['fan_in_src'] + 1e-6)
    features['fan_ratio_dst'] = features['fan_out_dst'] / (features['fan_in_dst'] + 1e-6)
    features['fan_total_src'] = features['fan_in_src'] + features['fan_out_src']
    features['fan_total_dst'] = features['fan_in_dst'] + features['fan_out_dst']
    
    # Connectivity features
    features['connection_asymmetry'] = np.abs(features['fan_total_src'] - features['fan_total_dst'])
    features['connection_intensity'] = features['fan_total_src'] * features['fan_total_dst']
    
    # IP class-based features
    features['ip_class_mismatch'] = (
         (features['src_ip_class_Class A'] & features['dst_ip_class_Class C']) |
         (features['src_ip_class_Class C'] & features['dst_ip_class_Class A'])
    ).astype(int)
    
    # Suspicious behavior detection based on IP classes
    features['potential_broadcast_attack'] = (
         features['dst_ip_class_Class D (multicast)'] & 
         (features['bidirectional_packets'] > features['bidirectional_packets'].quantile(0.95))
    ).astype(int)
    
    return features

def create_timing_features(df):
    """
    Create features based on timing characteristics
    """
    print("Creating timing features...")
    features = df.copy()
    
    # Timing ratios
    features['piat_ratio_src2dst'] = features['src2dst_mean_piat_ms'] / (features['dst2src_mean_piat_ms'] + 1e-6)
    features['piat_ratio_stddev'] = features['bidirectional_stddev_piat_ms'] / (features['bidirectional_mean_piat_ms'] + 1e-6)
    
    # Regularity features
    features['timing_regularity'] = 1 - (features['bidirectional_stddev_piat_ms'] / 
                                         (features['bidirectional_mean_piat_ms'] + 1e-6))
    
    # Burst detection
    features['burst_factor'] = (features['bidirectional_packets'] / 
                                (features['bidirectional_duration_ms'] + 1e-6))
    
    return features

def create_protocol_features(df):
    """
    Create features based on protocols and ports
    """
    print("Creating protocol features...")
    features = df.copy()
    
    # Suspicious protocol/port combinations
    features['suspicious_port_protocol'] = (
         (features['protocol_17'] & features['dst_port_class_WellKnown'] & 
          (features['bidirectional_packets'] < features['bidirectional_packets'].quantile(0.1)))
    ).astype(int)
    
    # Protocol anomalies
    features['protocol_anomaly'] = (
         (features['protocol_0'] | features['protocol_1'] | features['protocol_89'] | features['protocol_132'])
    ).astype(int)
    
    return features

def create_packet_features(df):
    """
    Create features based on packet characteristics
    """
    print("Creating packet features...")
    features = df.copy()
    
    # Packet size ratios
    features['ps_ratio_src2dst'] = features['src2dst_mean_ps'] / (features['dst2src_mean_ps'] + 1e-6)
    features['ps_variation_ratio'] = features['bidirectional_stddev_ps'] / (features['bidirectional_mean_ps'] + 1e-6)
    
    # Flow characteristics
    features['flow_efficiency'] = features['bidirectional_bytes'] / (features['bidirectional_packets'] + 1e-6)
    features['flow_regularity'] = 1 - (features['bidirectional_stddev_ps'] / (features['bidirectional_mean_ps'] + 1e-6))
    
    return features

def create_behavioral_features(df):
    """
    Create features for detecting specific behaviors
    """
    print("Creating behavioral features...")
    features = df.copy()
    
    # Scan detection
    features['potential_scan'] = (
         (features['fan_out_dst'] > features['fan_out_dst'].quantile(0.95)) &
         (features['bidirectional_packets'] < features['bidirectional_packets'].quantile(0.05)) &
         (features['bidirectional_duration_ms'] < features['bidirectional_duration_ms'].quantile(0.05))
    ).astype(int)
    
    # DDoS detection
    features['potential_ddos'] = (
         (features['fan_in_dst'] > features['fan_in_dst'].quantile(0.95)) &
         (features['bidirectional_packets'] > features['bidirectional_packets'].quantile(0.95)) &
         (features['bidirectional_mean_piat_ms'] < features['bidirectional_mean_piat_ms'].quantile(0.05))
    ).astype(int)
    
    # Data exfiltration detection
    features['potential_exfiltration'] = (
         (features['bidirectional_bytes'] > features['bidirectional_bytes'].quantile(0.95)) &
         (features['dst2src_mean_ps'] < features['src2dst_mean_ps'] * 0.1)
    ).astype(int)
    
    return features

def create_final_features(df):
    """
    Combine all features into a single DataFrame
    """
    print("Combining all feature sets...")
    connection_features = create_connection_pattern_features(df)
    timing_features = create_timing_features(df)
    protocol_features = create_protocol_features(df)
    packet_features = create_packet_features(df)
    behavioral_features = create_behavioral_features(df)
    
    final_features = pd.concat([
         connection_features,
         timing_features,
         protocol_features,
         packet_features,
         behavioral_features
    ], axis=1)
    
    # Remove duplicate columns
    final_features = final_features.loc[:, ~final_features.columns.duplicated()]
    
    return final_features

print("Generating enriched feature DataFrame...")
enriched_features_df = create_final_features(df)
# Save the DataFrame with labels to a CSV file
output_path = 'csv_files/final_features_flows.csv'
enriched_features_df.to_csv(output_path, index=False)
print(f"Final DataFrame saved to {output_path}")
