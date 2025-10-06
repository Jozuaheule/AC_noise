from data_processor import df
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sampling_frequency = 40000  # Hz
N_samples = len(df.index)
chunk_duration = 0.1
Pe_0 = 20e-5  # N/m^2

# Voeg een tijd-as toe (in seconden)
df["time"] = np.arange(N_samples) / sampling_frequency

def plot_spectrum(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["amplitude"], linewidth=0.7)
    plt.title("Sound Pressure vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Pa)")
    plt.xlim(left=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. Plot time vs. signal in figure 1
duration = N_samples / sampling_frequency
print(f"duration: {duration}")

df["samples"] = df["samples"] / sampling_frequency
# plot_spectrum(df)


# 3. How many samples correspond to a data lengt of T 0.1 seconds
samples_per_chunck = sampling_frequency * chunk_duration
print(f"samples_per_chunck: {samples_per_chunck}")


# 4. Calculate the so-called  ́effective sound pressure ́ as a function of time using equation 1.4 from the lecture notes
# x_2_data is a list containing the squared average pressure for each chunk

effective_pressures = []
fourier_data = {}  # Use a dictionary to store Fourier transform results

sample_iterations = int(np.ceil((N_samples/samples_per_chunck)))
x_effective_pres = np.arange(0.05, duration, chunk_duration)

for i in range(sample_iterations):
    start_idx = int(i * samples_per_chunck)
    end_idx = int(min(start_idx + samples_per_chunck, N_samples))

    chunk = df[start_idx:end_idx]  # Extract the chunk

    # Perform Fourier transformation
    fft_result = np.fft.fft(chunk['amplitude']**2)
    fft_freqs = np.fft.fftfreq(len(chunk), d=1/sampling_frequency)
    pos_mask = fft_freqs >= 0

    # Store results in the dictionary
    fourier_data[f'chunk_{i}'] = {
        'fft': fft_result[pos_mask],
        'freqs': fft_freqs[pos_mask]
    }

    # appending chunk to x_2_data set
    avg_x = np.average(chunk)
    effective_pres_dt = np.sqrt(1/chunk_duration*np.sum(chunk['amplitude']**2))

    effective_pressures.append(effective_pres_dt)

# Plot function for effective sound pressure
def plot_effective_pres(df, effective_pressures):

    plt.plot(df["samples"], df["amplitude"], linewidth=0.7, color='blue')
    plt.plot(x_effective_pres, effective_pressures, linewidth=0.7, color='green')

    plt.title("Spectrum of Sound Pressure")
    plt.xlabel("Time (s)")
    plt.xlim(left=0)    
    plt.ylabel("Sound Pressure (Pa)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_effective_pres(df, effective_pressures)


# 5. Calculate the OSPL for each chunk. 

OSPL = [10 * np.log10(pres**2 / Pe_0**2) for pres in effective_pressures]


def plot_OSPL(x_effective_pres, OSPL):

    plt.plot(x_effective_pres, OSPL, linewidth=0.7, color='red')

    plt.title("Overall Sound Pressure Level (OSPL)")
    plt.xlabel("Time (s)")
    plt.ylabel("OSPL (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_OSPL(x_effective_pres, OSPL)

# 6. DFT transformation

fourier_df = pd.DataFrame.from_dict(fourier_data, orient='index')
fourier_df = fourier_df[:-1]

print(fourier_df)

# Plot function for individual fourier transforms
def fourier_plot(fourier_df, x_effective_pres):

    plt.plot(fourier_df['freqs'][0], fourier_df['fft'][0] , linewidth=0.7, color='purple')
    plt.title("Fourier Transform")
    plt.xlabel("Frequency (s)")
    plt.ylabel("Pa^2/Hz (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
   
#fourier_plot(fourier_df)

# Convert list of complex FFT results to magnitudes
magnitudes = np.array([np.abs(np.array(f)) for f in fourier_df['fft']])

# Convert frequencies to a 2D array (assuming they’re all identical)
freqs = np.array(fourier_df['freqs'].iloc[0])/1000              #kHz transformation
times = x_effective_pres  # one per chunk

# Convert to decibels (dB scale)
spectrogram = 20 * np.log10(magnitudes + 1e-6)                  # NAKIJKEN

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.imshow(
    spectrogram.T,
    extent=[times.min(), times.max(), freqs.min(), freqs.max()],
    origin='lower',
    aspect='auto',
    cmap='jet',
    vmin=-20, vmax=60
)
plt.colorbar(label='dB')
plt.xlabel('time [s]')
plt.ylabel('frequency [kHz]')
plt.title('Spectrogram of flyover')
plt.show()
