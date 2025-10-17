import pandas as pd
import matplotlib.pyplot as plt

# Load data
excel_file = 'omas_history.xlsx'
df = pd.read_excel(excel_file)

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# First plot: Pulse length vs Plasma Current
axes[0].scatter(df['pulse_duration'], df['max_plasma_current'], alpha=0.7)
axes[0].set_xlabel('Pulse length [ms]')
axes[0].set_ylabel('Plasma Current $I_p$ at peak [kA]')

# Second plot: Toroidal Field vs Plasma Current
axes[1].scatter(df['toroidal_field'], df['max_plasma_current'], alpha=0.7)
axes[1].set_xlabel('Toroidal Field $B_t$ on axis [T]')
axes[1].set_ylabel('Plasma Current $I_p$ at peak [kA]')

plt.tight_layout()
plt.savefig('omas_history.png')
plt.show()

