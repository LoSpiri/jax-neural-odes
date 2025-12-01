import os
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class SepsisDataset:
    def __init__(self):
        pd.set_option("display.max_columns", None) 
        pd.set_option("display.width", 2000)
        pd.set_option("display.expand_frame_repr", False)
        
        cwd = os.getcwd()
        labels_file = os.path.join(cwd, 'labels.csv') 
        time_series_dir = os.path.join(cwd, 'time_series')
        
        self.patients = []
        self.outcomes = []
        with open(labels_file, 'r') as f:
            # skip header
            next(f)
            for line in f:
                parts = line.strip().split(',')
                self.patients.append(parts[0])
                if parts[1] == "0.0":
                    self.outcomes.append(False)
                elif parts[1] == "1.0":
                    self.outcomes.append(True)
                else:
                    raise ValueError(f"Unexpected outcome value: {parts[1:]}")
        
        self.column_names =  [
            'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
            'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
            'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
            'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
            'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
            'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 
            'HospAdmTime', 'ICULOS'
        ]
        self.demographic_columns = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
        
        self.time_series_data = {}
        for filename in tqdm(os.listdir(time_series_dir)):
            if filename.endswith('.csv'):
                file_path = os.path.join(time_series_dir, filename)
                df = pd.read_csv(file_path, header=None, names=self.column_names)
                patient = filename[:-4]
                self.time_series_data[patient] = df
                
    def _plot_distributions_for_each_column_with_outcomes(self):
        num_columns = len(self.column_names)
        num_rows = (num_columns + 3) // 4  
        
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 4))
        axes = axes.flatten()
        
        for i, column in enumerate(self.column_names):
            all_values_0 = pd.concat([
                df[column] for patient, outcome in zip(self.patients, self.outcomes) 
                if not outcome for df in [self.time_series_data[patient]]
            ])
            all_values_1 = pd.concat([
                df[column] for patient, outcome in zip(self.patients, self.outcomes) 
                if outcome for df in [self.time_series_data[patient]]
            ])
            axes[i].hist(all_values_0.dropna(), bins=50, color='blue', alpha=0.5, label='Outcome 0')
            axes[i].hist(all_values_1.dropna(), bins=50, color='red', alpha=0.5, label='Outcome 1')
            axes[i].set_title(f'Distribution of {column} by Outcome')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
    def bar_plot_of_missing_values(self):
        missing_counts = {col: 0 for col in self.column_names}
        total_counts = {col: 0 for col in self.column_names}
        for df in self.time_series_data.values():
            n = len(df)
            for col in self.column_names:
                if col in df.columns:
                    missing_counts[col] += df[col].isna().sum()
                    total_counts[col] += n

        missing_pct = {
            col: (missing_counts[col] / total_counts[col] * 100) if total_counts[col] else 0.0
            for col in self.column_names
        }
        present_pct = {col: 100.0 - missing_pct[col] for col in self.column_names}

        sort_by = "missing"  # options: "missing", "present", or None
        if sort_by == "missing":
            order = sorted(self.column_names, key=lambda c: missing_pct[c], reverse=True)
        elif sort_by == "present":
            order = sorted(self.column_names, key=lambda c: present_pct[c], reverse=True)
        else:
            order = list(self.column_names)

        missing_vals = [missing_pct[c] for c in order]
        present_vals = [present_pct[c] for c in order]

        plt.figure(figsize=(16, 7))
        plt.bar(order, present_vals, label="Present (%)")                   # base fill
        plt.bar(order, missing_vals, bottom=present_vals, label="Missing (%)")  # stacked missing
        plt.xticks(rotation=90)
        plt.ylim(0, 100)
        plt.ylabel("Share of records (%)")
        plt.title("Data Completeness per Feature (Present vs Missing)")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend()

        for i, (p, m) in enumerate(zip(present_vals, missing_vals)):
            if m > 0:
                plt.text(i, p + m/2, f"{m:.1f}%", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.show()
        
    def boxplots_columns_and_outcomes(self):
        for col in self.column_names:
            data_0 = []
            data_1 = []
            for patient, outcome in zip(self.patients, self.outcomes):
                df = self.time_series_data[patient]
                if not outcome:
                    data_0.extend(df[col].dropna().tolist())
                else:
                    data_1.extend(df[col].dropna().tolist())
            plt.figure(figsize=(8, 6))
            plt.boxplot([data_0, data_1], labels=['Outcome 0', 'Outcome 1'], showfliers=False)
            plt.title(f'{col}')
            plt.ylabel(col)
            plt.show()