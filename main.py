from data_processor import df
import numpy as np


sampling_frequency = 40000
N_samples = len(df.index)
chunk_length = 0.1

# 1. Load aircraft noise data file 


# 2. Plot time vs. signal in figure 1
duration = N_samples / sampling_frequency
print(f"duration: {duration}")


# 3. How many samples correspond to a data lengt of T 0.1 seconds
samples_per_chunck = sampling_frequency * chunk_length
print(f"samples_per_chunck: {samples_per_chunck}")


# 4. Calculate the so-called  ́effective sound pressure ́ as a function of time using equation 1.4 from the lecture notes
# basis staat maar is nog niet helemaal af

x_2_data = []

iterations = int(np.ceil((N_samples/samples_per_chunck)))
print(iterations)

for i in range(iterations):
    start_idx = int(i * samples_per_chunck)
    end_idx = int(min(start_idx + samples_per_chunck, N_samples))
    chunk = df[start_idx:end_idx]
    
    # Calculating effective pressure

    avg_x = np.average(chunk)
    x_2_data.append(avg_x**2)

effective_pres = 1/duration*sum(x_2_data)