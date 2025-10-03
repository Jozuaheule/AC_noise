import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

# Load .mat file
mat_file = scipy.io.loadmat("Flyover_No_5.mat")


variables = mat_file.keys()
print(variables)

# Extract the variable you care about
data = mat_file["sound_pressure"]

# Convert to DataFrame
# Convert to DataFrame (flatten to 1D)
df = pd.DataFrame(data.flatten(), columns=["amplitude"])

# Make an index column representing frequency bins
df["frequency_bin"] = df.index                          # x = df["frequency_bin"], y = df["amplitude"]

if '__main__':

    # Plot spectrum
    plt.figure(figsize=(12,6))
    plt.plot(df["frequency_bin"], df["amplitude"], linewidth=0.7)

    plt.title("Spectrum of Sound Pressure")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
