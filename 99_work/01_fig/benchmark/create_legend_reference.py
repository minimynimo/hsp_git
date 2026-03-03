import matplotlib.pyplot as plt
import numpy as np

# Color and style definitions from the main script
color_list = {
    "RF":   "#FF7F00",   # Orange
    "XGB":  "#9C4803",   # Dark Orange
    "TCN":  "#E41A1C",   # Red
    "LSTM": "#9808B2"    # Purple
}

marker_list = {
    "regional": "o",     # Circle
    "individual": "s",   # Square
    "adapted": "^"       # Triangle
}

linestyle_list = {
    "regional": "-",     # Solid
    "individual": "--",  # Dashed
    "adapted": ":"       # Dotted
}

model_list = ["RF", "XGB", "TCN", "LSTM"]
train_types = ["regional", "individual", "adapted"]

# Create a legend reference plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create sample data for demonstration
x = np.linspace(0, 10, 100)

y_offset = 0
for model in model_list:
    for train_type in train_types:
        y = np.sin(x + y_offset) + y_offset
        ax.plot(x, y, 
               color=color_list[model],
               linestyle=linestyle_list[train_type],
               marker=marker_list[train_type],
               markersize=6,
               markevery=10,
               label=f"{model}_{train_type}",
               linewidth=2)
        y_offset += 0.5

ax.set_xlabel('X Value')
ax.set_ylabel('Y Value')
ax.set_title('Color and Style Reference Guide\nColors by Model, Markers/Lines by Training Type')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/data0/funato/hsp2/0_out/1_fig/benchmark/ML_training_comparison/legend_reference.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print("Legend reference guide created!")