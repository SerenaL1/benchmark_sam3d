import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Setup Data
group_type = "AMD"
data = {
    "Metric": ["F1@0.01", "Voxel-IoU", "Voxel-Dice", "Chamfer", "EMD"],
    "Energy Dist": [0.079545, 0.076666, 0.091803, 0.013430, 0.011472],
    "Avg Cross": [0.8944, 0.8855, 0.8024, 0.1142, 0.1232],
    "Avg Intra-A": [0.8258, 0.8143, 0.7061, 0.0768, 0.0956],
    "Avg Intra-B": [0.8835, 0.8801, 0.8070, 0.1382, 0.1393]
}
output_file = f'energy_distance_{group_type}.png'

df = pd.DataFrame(data)

# Split into groups
group1_metrics = ["F1@0.01", "Voxel-IoU", "Voxel-Dice"]
df_g1 = df[df["Metric"].isin(group1_metrics)].copy()

group2_metrics = ["Chamfer", "EMD"]
df_g2 = df[df["Metric"].isin(group2_metrics)].copy()

# 2. Nature Portfolio Style Configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11  # Slightly smaller for better fit
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'

# Colors
colors = {
    "Avg Intra-A": "#0072B2",  # Deep Ocean Blue
    "Avg Cross": "#009E73",    # Teal Green
    "Avg Intra-B": "#D55E00",  # Vermillion Orange
    "Energy Dist": "#CC79A7"   # Rose Purple
}

# 3. Create Figure using Constrained Layout (Fixes spacing issues automatically)
fig = plt.figure(figsize=(13, 8), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])

def format_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='y', direction='out', width=1.2)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

def autolabel(rects, ax, fmt='{:.3f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2C3E50')

def add_panel_label(ax, label):
    # Adds 'a', 'b', etc. to the top left corner (Nature style)
    ax.text(-0.05, 1.1, label, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='right')

# --- Panel A: Components (Similarity) ---
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(df_g1))
width = 0.25

rects1 = ax1.bar(x - width, df_g1["Avg Intra-A"], width, label="Intra-Normal", 
                 color=colors["Avg Intra-A"], alpha=0.9, edgecolor='white')
rects2 = ax1.bar(x, df_g1["Avg Cross"], width, label=f"Cross (Normal vs {group_type})", 
                 color=colors["Avg Cross"], alpha=0.9, edgecolor='white')
rects3 = ax1.bar(x + width, df_g1["Avg Intra-B"], width, label=f"Intra-{group_type}", 
                 color=colors["Avg Intra-B"], alpha=0.9, edgecolor='white')

ax1.set_ylabel('Mean Score', fontweight='bold')
ax1.set_title('Distribution Components (Similarity Scores)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df_g1["Metric"])

# Dynamic Y-Limit to prevent overlapping text
max_h = max(df_g1[["Avg Intra-A", "Avg Cross", "Avg Intra-B"]].max())
ax1.set_ylim(0, max_h * 1.35) 

# Legend inside to avoid title overlap
ax1.legend(frameon=False, loc='upper center', ncol=1, fontsize=9) 
format_ax(ax1)
add_panel_label(ax1, 'a')

autolabel(rects1, ax1, '{:.2f}')
autolabel(rects2, ax1, '{:.2f}')
autolabel(rects3, ax1, '{:.2f}')

# --- Panel B: Components (Distance) ---
ax2 = fig.add_subplot(gs[0, 1])
x2 = np.arange(len(df_g2))

rects1_2 = ax2.bar(x2 - width, df_g2["Avg Intra-A"], width, label="Intra-Normal", 
                   color=colors["Avg Intra-A"], alpha=0.9, edgecolor='white')
rects2_2 = ax2.bar(x2, df_g2["Avg Cross"], width, label=f"Cross (Normal vs {group_type})", 
                   color=colors["Avg Cross"], alpha=0.9, edgecolor='white')
rects3_2 = ax2.bar(x2 + width, df_g2["Avg Intra-B"], width, label=f"Intra-{group_type}", 
                   color=colors["Avg Intra-B"], alpha=0.9, edgecolor='white')

ax2.set_ylabel('Mean Distance', fontweight='bold')
ax2.set_title('Distribution Components (Distance)', fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(df_g2["Metric"])

# Dynamic Y-Limit
max_h2 = max(df_g2[["Avg Intra-A", "Avg Cross", "Avg Intra-B"]].max())
ax2.set_ylim(0, max_h2 * 1.35)

ax2.legend(frameon=False, loc='upper center', ncol=1, fontsize=9)
format_ax(ax2)
add_panel_label(ax2, 'b')

autolabel(rects1_2, ax2, '{:.3f}')
autolabel(rects2_2, ax2, '{:.3f}')
autolabel(rects3_2, ax2, '{:.3f}')

# --- Panel C: Energy Dist (Similarity) ---
ax3 = fig.add_subplot(gs[1, 0])
rects_e1 = ax3.bar(df_g1["Metric"], df_g1["Energy Dist"], width=0.4, 
                   color=colors["Energy Dist"], alpha=0.9, edgecolor='white')
ax3.set_ylabel('Energy Distance', fontweight='bold')
ax3.set_title('Energy Distance (Similarity)', fontweight='bold')

max_he1 = df_g1["Energy Dist"].max()
ax3.set_ylim(0, max_he1 * 1.35)

format_ax(ax3)
add_panel_label(ax3, 'c')
autolabel(rects_e1, ax3, '{:.4f}')

# --- Panel D: Energy Dist (Distance) ---
ax4 = fig.add_subplot(gs[1, 1])
rects_e2 = ax4.bar(df_g2["Metric"], df_g2["Energy Dist"], width=0.4, 
                   color=colors["Energy Dist"], alpha=0.9, edgecolor='white')
ax4.set_ylabel('Energy Distance', fontweight='bold')
ax4.set_title('Energy Distance (Distance)', fontweight='bold')

max_he2 = df_g2["Energy Dist"].max()
ax4.set_ylim(0, max_he2 * 1.35)

format_ax(ax4)
add_panel_label(ax4, 'd')
autolabel(rects_e2, ax4, '{:.4f}')

# Note: No need for manual subplots_adjust when using layout="constrained"
plt.savefig(output_file, format='svg', bbox_inches='tight')
plt.show()