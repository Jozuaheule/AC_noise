import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Load .mat file
mat_file = scipy.io.loadmat("Flyover_No_5.mat")

variables = mat_file.keys()
print(f"variables: {variables}")

# Extract the variable you care about
data = mat_file["sound_pressure"]

# Convert to DataFrame (flatten to 1D)
df = pd.DataFrame(data.flatten(), columns=["amplitude"])

# Make an index column representing frequency bins
df["samples"] = df.index                          # x = df["samples"], y = df["amplitude"]


if '__main__':
    # Save to text file (tab-separated)
    #df.to_csv("spectrum_data.txt", sep="\t", index=False)

    # Plot spectrum
    plt.figure(figsize=(12,6))
    plt.plot(df["samples"], df["amplitude"], linewidth=0.7)

    plt.title("Spectrum of Sound Pressure")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
