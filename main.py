from data_processor import df
import numpy as np
import matplotlib.pyplot as plt


sampling_frequency = 40000
N_samples = len(df.index)
chunk_duration = 0.1

# 1. Load aircraft noise data file 

def plot_spectrum(df):
    plt.figure(figsize=(12,6))
    plt.plot(df["samples"], df["amplitude"], linewidth=0.7)

    plt.title("Spectrum of Sound Pressure")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. Plot time vs. signal in figure 1
duration = N_samples / sampling_frequency

df["samples"] = df["samples"] / sampling_frequency
#plot_spectrum(df)


# 3. How many samples correspond to a data lengt of T 0.1 seconds
samples_per_chunck = sampling_frequency * chunk_duration
print(f"samples_per_chunck: {samples_per_chunck}")


# 4. Calculate the so-called  ́effective sound pressure ́ as a function of time using equation 1.4 from the lecture notes
# x_2_data is a list containing the squared average pressure for each chunk

effective_pressures = []

sample_iterations = int(np.ceil((N_samples/samples_per_chunck)))

for i in range(sample_iterations):
    start_idx = int(i * samples_per_chunck)
    end_idx = int(min(start_idx + samples_per_chunck, N_samples))
    #print(f"start_idx: {start_idx}, end_idx: {end_idx}")

    chunk = df[start_idx:end_idx]           # x(t) waardes van de chunk

    # appending chunk to x_2_data set
    avg_x = np.average(chunk)
    effective_pres_dt = np.sqrt(1/chunk_duration*np.sum(chunk['amplitude']**2))

    effective_pressures.append(effective_pres_dt)

# Plot function for effective sound pressure
def plot_effective_pres(df, effective_pressures):
    
    x_effective_pres = np.arange(0.05, duration, chunk_duration)

    plt.plot(df["samples"], df["amplitude"], linewidth=0.7, color='blue')
    plt.plot(x_effective_pres, effective_pressures, linewidth=0.7, color='green')

    plt.title("Spectrum of Sound Pressure")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_effective_pres(df, effective_pressures)





