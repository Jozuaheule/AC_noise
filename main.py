df = pd.read_csv("spectrum_dataframe.csv")

sampling_frequency = 40000
N_samples = 1086250
chunk_length = 0.1


# 1. Load aircraft noise data file 



# 2. Plot time vs. signal in figure 1
duration = N_samples / sampling_frequency
print(f"duration: {duration}")


# 3. How many samples correspond to a data lengt of T 0.1 seconds
samples_per_chunck = sampling_frequency * chunk_length
print(f"samples_per_chunck: {samples_per_chunck}")


