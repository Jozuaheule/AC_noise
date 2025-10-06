from data_processor import df
import numpy as np

sampling_frequency = 40000
N_samples = len(df.index)
chunk_duration = 0.1

# 1. Load aircraft noise data file 


# 2. Plot time vs. signal in figure 1
duration = N_samples / sampling_frequency
print(f"duration: {duration}")


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
    effective_pres_dt = 1/chunk_duration*np.sum(chunk['amplitude']**2)

    effective_pressures.append(effective_pres_dt)

