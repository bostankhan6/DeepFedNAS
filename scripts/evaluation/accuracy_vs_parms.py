import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
# Define the paths to your CSV files
# Make sure these paths are correct for your environment
DEEPFEDNAS_CSV_PATH = 'results/deepfednas_subnet_details_cifar100_correct_cached_60_subnets_cifar100.csv'
SUPERFEDNAS_CSV_PATH = 'results/cifar-100-baseline_results_macs_parameters_10_pnts/subnet_details_cifar100_correct_baseline_cifar100.csv'
OUTPUT_DIR = 'results/parameter_analyses/plots_and_tables' # Directory to save generated files

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Loading Data ---")
try:
    df_deepfednas = pd.read_csv(DEEPFEDNAS_CSV_PATH)
    df_superfednas = pd.read_csv(SUPERFEDNAS_CSV_PATH)
    print(f"Loaded DeepFedNAS data: {len(df_deepfednas)} rows")
    print(f"Loaded SuperFedNAS data: {len(df_superfednas)} rows")
except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the paths are correct.")
    exit()

# --- Data Preprocessing ---
print("\n--- Preprocessing Data ---")
# Add a 'Method' column for easy grouping and plotting
df_deepfednas['Method'] = 'DeepFedNAS'
df_superfednas['Method'] = 'SuperFedNAS'

# Combine the two DataFrames
df_combined = pd.concat([df_deepfednas, df_superfednas], ignore_index=True)

# Convert num_parameters to Millions for better readability in plots/tables
df_combined['num_parameters_M'] = df_combined['num_parameters'] / 1e6
# Convert actual_macs to Billions
df_combined['actual_macs_B'] = df_combined['actual_macs'] / 1e9

print("Combined DataFrame head:")
print(df_combined.head())
print("\nCombined DataFrame info:")
df_combined.info()

# --- Generate Scatter Plot: Test Accuracy vs. Number of Parameters ---
print("\n--- Generating Plot: Test Accuracy vs. Number of Parameters ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_combined,
    x='num_parameters_M',
    y='test_accuracy',
    hue='Method',
    style='Method',
    s=100, # size of points
    alpha=0.7,
    edgecolor='w', # white edge for better visibility
    linewidth=0.5
)

# Apply log scale to x-axis if parameters range widely
# Check the range and decide if log scale is appropriate
min_params = df_combined['num_parameters_M'].min()
max_params = df_combined['num_parameters_M'].max()
if (max_params / min_params) > 10: # If max is more than 10x min, log scale is usually good
    plt.xscale('log')
    plt.xlabel('Number of Parameters (Millions, Log Scale)')
else:
    plt.xlabel('Number of Parameters (Millions)')

plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs. Model Parameters for DeepFedNAS and SuperFedNAS Subnets (CIFAR-100)')
plt.grid(True, which="both", ls="--", c='0.7')
plt.legend(title='Method')
plt.tight_layout()

plot_filename = os.path.join(OUTPUT_DIR, 'param_accuracy_tradeoff_cifar100.png')
plt.savefig(plot_filename, dpi=300)
print(f"Plot saved to: {plot_filename}")
plt.show() # Display the plot


# --- Prepare Data for Summary Table ---
print("\n--- Preparing Data for Summary Table ---")

# Group by method and the original macs_target_bin for consistent aggregation
# Using macs_constraint for grouping, then averaging actual_macs_B for presentation
table_data = df_combined.groupby(['macs_constraint', 'Method']).agg(
    Avg_MACs_B=('actual_macs_B', 'mean'),
    Avg_Accuracy=('test_accuracy', 'mean'),
    Std_Dev_Accuracy=('test_accuracy', 'std'),
    Avg_Parameters_M=('num_parameters_M', 'mean')
).reset_index()

# Sort by macs_constraint to ensure table order is logical
table_data = table_data.sort_values(by='macs_constraint')

# Calculate Parameter Reduction (%) for DeepFedNAS relative to SuperFedNAS
final_table_rows = []
for mac_cons in table_data['macs_constraint'].unique():
    subset = table_data[table_data['macs_constraint'] == mac_cons]

    deepfednas_row = subset[subset['Method'] == 'DeepFedNAS'].iloc[0] if 'DeepFedNAS' in subset['Method'].values else None
    superfednas_row = subset[subset['Method'] == 'SuperFedNAS'].iloc[0] if 'SuperFedNAS' in subset['Method'].values else None

    param_reduction = np.nan
    if deepfednas_row is not None and superfednas_row is not None and superfednas_row['Avg_Parameters_M'] > 0:
        param_reduction = ((superfednas_row['Avg_Parameters_M'] - deepfednas_row['Avg_Parameters_M']) / superfednas_row['Avg_Parameters_M']) * 100
    
    # Append DeepFedNAS row
    if deepfednas_row is not None:
        final_table_rows.append({
            'MACs Target (B)': f"{mac_cons/1e9:.2f}B (Max)", # Indicate it's the upper limit
            'Method': 'DeepFedNAS',
            'Avg. MACs (B)': f"{deepfednas_row['Avg_MACs_B']:.2f}",
            'Avg. Accuracy (%)': f"{deepfednas_row['Avg_Accuracy']:.2f}",
            'Std. Dev. Acc. (%)': f"{deepfednas_row['Std_Dev_Accuracy']:.2f}",
            'Avg. Parameters (M)': f"{deepfednas_row['Avg_Parameters_M']:.2f}",
            'Param. Reduction (%)': f"{param_reduction:.2f}" if not np.isnan(param_reduction) else '-'
        })
    
    # Append SuperFedNAS row
    if superfednas_row is not None:
        final_table_rows.append({
            'MACs Target (B)': f"{mac_cons/1e9:.2f}B (Max)",
            'Method': 'SuperFedNAS',
            'Avg. MACs (B)': f"{superfednas_row['Avg_MACs_B']:.2f}",
            'Avg. Accuracy (%)': f"{superfednas_row['Avg_Accuracy']:.2f}",
            'Std. Dev. Acc. (%)': f"{superfednas_row['Std_Dev_Accuracy']:.2f}",
            'Avg. Parameters (M)': f"{superfednas_row['Avg_Parameters_M']:.2f}",
            'Param. Reduction (%)': '-' # Not applicable for baseline
        })

df_final_table = pd.DataFrame(final_table_rows)

table_filename = os.path.join(OUTPUT_DIR, 'param_accuracy_summary_table_cifar100.csv')
df_final_table.to_csv(table_filename, index=False)
print(f"Summary table saved to: {table_filename}")

# Print the final table for immediate review
print("\n--- Final Summary Table (for LaTeX/Markdown conversion) ---")
print(df_final_table.to_string(index=False)) # Use to_string to print full DataFrame without truncation

print("\nAnalysis complete. Check the 'plots_and_tables' directory for outputs.")