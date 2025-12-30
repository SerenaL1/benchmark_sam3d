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
ax4_ylim = [0.008, 0.016]

group_type = "DR"
data = {
    "Metric": ["F1@0.01", "Voxel-IoU", "Voxel-Dice", "Chamfer", "EMD"],
    "Energy Dist": [0.053474, 0.051260, 0.053550, 0.003352, 0.006452],
    "Avg Cross": [0.8660, 0.8593, 0.7614, 0.0996, 0.1206],
    "Avg Intra-A": [0.8289, 0.8171, 0.7100, 0.0774, 0.0956],
    "Avg Intra-B": [0.8495, 0.8502, 0.7593, 0.1185, 0.1393]
}
ax4_ylim = [0.0, 0.008]

group_type = "Glaucoma"
data = {
    "Metric": ["F1@0.01", "Voxel-IoU", "Voxel-Dice", "Chamfer", "EMD"],
    "Energy Dist": [0.072662, 0.066567, 0.076171, 0.017622, 0.024695],
    "Avg Cross": [0.8781, 0.8711, 0.7791, 0.1156, 0.1333],
    "Avg Intra-A": [0.8267, 0.8155, 0.7079, 0.0771, 0.0957],
    "Avg Intra-B": [0.8568, 0.8601, 0.7742, 0.1364, 0.1463]
}
ax4_ylim = [0.01, 0.026]

output_file = f'data4paper/energy_distance_{group_type}.svg'

df = pd.DataFrame(data)

# Split into groups
group1_metrics = ["F1@0.01", "Voxel-IoU", "Voxel-Dice"]
df_g1 = df[df["Metric"].isin(group1_metrics)]

group2_metrics = ["Chamfer", "EMD"]
df_g2 = df[df["Metric"].isin(group2_metrics)]

# 2. Nature Portfolio Style Configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none' 

# Enhanced Modern Scientific Palette
# Inspired by Nature/Science journals with better contrast and visual appeal
colors = {
    "Avg Intra-A": "#0072B2",  # Deep Ocean Blue (excellent for colorblind)
    "Avg Cross": "#009E73",    # Teal Green (professional, stands out)
    "Avg Intra-B": "#D55E00",  # Vermillion Orange (warm, high contrast)
    "Energy Dist": "#CC79A7"   # Rose Purple (elegant, distinct)
}

# 3. Create Figure and Grid
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.4, wspace=0.2)

def format_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='y', direction='out', width=1.5)
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
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2C3E50')

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

ax1.set_ylabel('Mean Score') # fontweight='bold'
ax1.set_title('Distribution Components (Similarity Scores)', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(df_g1["Metric"])
ax1.set_ylim(0, 1.15)
ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3)
format_ax(ax1)

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

ax2.set_ylabel('Mean Distance') # fontweight='bold'
ax2.set_title('Distribution Components (Distance Metrics)', fontweight='bold', pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(df_g2["Metric"])
ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3)
ax2.set_ylim(0, 0.17)
format_ax(ax2)

autolabel(rects1_2, ax2, '{:.3f}')
autolabel(rects2_2, ax2, '{:.3f}')
autolabel(rects3_2, ax2, '{:.3f}')

# --- Panel C: Energy Dist (Similarity) ---
ax3 = fig.add_subplot(gs[1, 0])
rects_e1 = ax3.bar(df_g1["Metric"], df_g1["Energy Dist"], width=0.3, 
                   color=colors["Energy Dist"], alpha=0.85, edgecolor='white', linewidth=1.5)
ax3.set_ylabel('Energy Distance')
ax3.set_title('Energy Distance (Similarity Scores)', fontweight='bold', pad=10)
ax3.set_ylim(0, 0.11)
format_ax(ax3)
autolabel(rects_e1, ax3, '{:.4f}')

# --- Panel D: Energy Dist (Distance) ---
ax4 = fig.add_subplot(gs[1, 1])
rects_e2 = ax4.bar(df_g2["Metric"], df_g2["Energy Dist"], width=0.3, 
                   color=colors["Energy Dist"], alpha=0.85, edgecolor='white', linewidth=1.5)
ax4.set_ylabel('Energy Distance')
ax4.set_title('Energy Distance (Distance Metrics)', fontweight='bold', pad=10)
# ax4.set_ylim(0, 0.016)
ax4.set_ylim(ax4_ylim[0], ax4_ylim[1])
format_ax(ax4)
autolabel(rects_e2, ax4, '{:.4f}')

fig.suptitle(f"Energy Distance between Groups Normal vs. {group_type}", fontsize=20, fontweight='bold', y=0.94)
plt.subplots_adjust(top=0.85)

# Save as SVG with tight bounding box
plt.savefig(output_file, format='svg', bbox_inches='tight')
plt.show()