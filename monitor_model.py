import pandas as pd
import os
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset

def generate_monitoring_report():
    print("--- Starting Evidently AI Monitoring ---")
    
    reference_file = 'Dataset/master_dataset.csv'
    current_file = 'prediction_logs.csv'
    
    # 1. Load Reference Data (Original training data)
    if not os.path.exists(reference_file):
        print(f"Error: Reference data {reference_file} not found.")
        return
    
    reference_data = pd.read_csv(reference_file)
    print(f"Loaded {len(reference_data)} reference samples.")
    
    # 2. Load Current Data (Logged predictions)
    if not os.path.exists(current_file):
        print(f"Warning: {current_file} not found. Creating a simulation for demonstration...")
        # Simulate some data if no logs exist yet
        current_data = reference_data.sample(min(500, len(reference_data)))
        # Introduce some "drift" by changing some labels 
        current_data.loc[current_data.index[:50], 'text'] = "This video is so bad and I hate it" 
        current_data.loc[current_data.index[:50], 'label'] = 'anger'
    else:
        current_data = pd.read_csv(current_file)
        print(f"Loaded {len(current_data)} current prediction samples.")
    
    # 3. Create Evidently Report
    # We monitor 'text' for data drift and 'label' for target drift
    print("Generating Drift Report...")
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset()
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # 4. Save to HTML
    output_path = 'monitoring_report.html'
    report.save_html(output_path)
    
    print(f"Reports saved to {output_path}")

if __name__ == "__main__":
    generate_monitoring_report()
