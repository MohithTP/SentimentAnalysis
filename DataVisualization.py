import pandas as pd
from ydata_profiling import ProfileReport

train_df = pd.read_csv("Dataset/master_dataset.csv")
logs_df = pd.read_csv("prediction_logs.csv")

profile = ProfileReport(train_df, minimal = True, title="Data Analysis Report")

#profile.to_notebook_iframe()

train_report = ProfileReport(train_df, title="Train Data") 
log_report = ProfileReport(logs_df, title="Live Logs")

comparison_report = train_report.compare(log_report)
comparison_report.to_file("comparison.html")

profile.to_file("Data_report.html")