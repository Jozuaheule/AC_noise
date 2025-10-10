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
    plt.ylabel("Pressure (Pa)")
    plt.xlim(left=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. Plot time vs. signal in figure 1
duration = N_samples / sampling_frequency
dt = 1/sampling_frequency
print(f"duration: {duration}")


df["samples"] = df["samples"] / sampling_frequency
#plot_spectrum(df)


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

    ## ----------- Calculations for Fourier vraag --------------- ## 
    # Perform Fourier transformation
    fft_result = np.fft.fft(chunk['amplitude']**2)          # Result is Xm
    fft_freqs = np.fft.fftfreq(len(chunk), d=dt)            # x as for frequency

    power_spectrum = (abs(fft_result)**2*dt**2)/chunk_duration

    # cut of all negative fourier frequencies
    pos_mask = fft_freqs >= 0

    # Store results in the dictionary
    fourier_data[f'chunk_{i}'] = {
        'fft': fft_result[pos_mask],
        'freqs': fft_freqs[pos_mask],
        'power': power_spectrum[pos_mask]
    }
  
    # for question 4
    effective_pres_dt = np.sqrt(1/chunk_duration*(np.sum(chunk['amplitude']**2))*dt)
    effective_pressures.append(effective_pres_dt)

# Plot function for effective sound pressure
def plot_effective_pres(df, effective_pressures):

    plt.plot(df["samples"], df["amplitude"], linewidth=0.7, color='blue')
    plt.plot(x_effective_pres, effective_pressures, linewidth=2, color='green')

    plt.title("Waveform with effective pressure")
    plt.xlabel("Time (s)")
    plt.xlim(left=0)    
    plt.ylabel("Sound Pressure (Pa)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_effective_pres(df, effective_pressures)


# 5. Calculate the OSPL for each chunk. 

OSPL = [10 * np.log10(pres**2 / Pe_0**2) for pres in effective_pressures]   # Moet de 10 * niet 20 * zijn?


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

# Understanding the data
#print(fourier_df['fft'])

"""
chunk_0      [(7.618810670341817+0j), (2.5961177079897713-0...
chunk_1      [(10.843702784360627+0j), (2.02737337200835+2....
chunk_2      [(8.417674759189797+0j), (3.281377089516819+1....
chunk_3      [(11.979303359124005+0j), (2.0629742929438426-...
chunk_4      [(4.7868172339375255+0j), (0.8881052325047121-...
                                   ...                        
chunk_266    [(18.57805760380159+0j), (-3.021973305277326-2...
chunk_267    [(22.260847107748965+0j), (-7.4790554924556485...
chunk_268    [(15.386660715917653+0j), (2.7581035562995044-...
chunk_269    [(19.35311567073333+0j), (1.1943722601702453-5...
chunk_270    [(16.504125052111046+0j), (1.490936037975648-1...
Name: fft, Length: 271, dtype: object
"""
#print(fourier_df['freqs'])

"""

chunk_0      [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_1      [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_2      [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_3      [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_4      [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
                                   ...                        
chunk_266    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_267    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_268    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_269    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
chunk_270    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0...
Name: freqs, Length: 271, dtype: object
"""
#print(fourier_df['power']) #>> shape (271, 2000) niet gehaald, dus omzetten
"""
chunk_0      [3.6278922519071447e-07, 4.256673053391928e-08...
chunk_1      [7.3491181297219e-07, 5.810210419277293e-08, 8...
chunk_2      [4.428578021968813e-07, 7.67921889481616e-08, ...
chunk_3      [8.968981810619978e-07, 1.205329125980937e-07,...
chunk_4      [1.4321012019450813e-07, 1.1016604407759497e-0...
                                   ...                        
chunk_266    [2.1571514020635626e-06, 8.298396698210809e-08...
chunk_267    [3.0971582122160965e-06, 3.5531945856409734e-0...
chunk_268    [1.479683299917272e-06, 8.575110688578846e-08,...
chunk_269    [2.3408942885299e-06, 1.746973383068544e-07, 2...
chunk_270    [1.7024133983482463e-06, 2.6855466033158217e-0...
Name: power, Length: 271, dtype: object
"""

lists = fourier_df['power'].tolist()
power_array = np.vstack(lists)

print(power_array)


# # Step 2: Stack them into a 2D numpy array
# power_array = np.vstack(lists)

# power_array = np.array()

# for i in len(fourier_df['power']):
#     power_array.vstack(i)

# print(power_array)

# print(fourier_df)

# Plot function for individual fourier transforms
def fourier_plot(fourier_df):

    plt.plot(fourier_df['freqs'][0], fourier_df['fft'][0] , linewidth=0.7, color='purple')
    plt.title("Fourier Transform")
    plt.xlabel("Frequency (s)")
    plt.ylabel("Pa^2/Hz (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
   
#fourier_plot(fourier_df)










# # Convert list of complex FFT results to magnitudes
# powers = np.array([np.array(p) for p in fourier_df['power']])         #Pm p114 reader
# fouriers = np.array([np.abs(np.array(f)) for f in fourier_df['fft']])

# # Convert frequencies to a 2D array (assuming they’re all identical)
# freqs = np.array(fourier_df['freqs'].iloc[0])/1000              #kHz transformation
# times = x_effective_pres  # one per chunk

# # Convert to decibels (dB scale)

# spectrogram = 20 * np.log10(powers)                  # NAKIJKEN
# print(spectrogram.shape)

# --- Plot ---
def spectrum_plot(spectrogram, freqs, times):

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

#spectrum_plot(spectrogram, freqs, times)


# 7  instantaneous OSPL (in dB) as a function of time
linear_matrix = 10 ** (spectrogram / 10)
OSPL_freq = 10 * np.log10(np.sum(linear_matrix, axis=1))
#print(spectrogram)
#print('OSPL', OSPL_freq.shape)
plt.figure(figsize=(10, 5))
plt.plot(x_effective_pres, OSPL, color='red', label='Time-domain OSPL (Q5)')
plt.plot(x_effective_pres[:len(OSPL_freq)], OSPL_freq, '--', color='blue', label='Freq-domain OSPL (Q7)')
plt.xlabel('Time (s)')
plt.ylabel('OSPL (dB)')
plt.title('Instantaneous OSPL Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()
