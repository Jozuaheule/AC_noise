import scipy.io
import pandas as pd

# Load .mat file
mat_file = scipy.io.loadmat("Flyover_No_5.mat")


variables = mat_file.keys()
print(variables)

# Extract the variable you care about
data = mat_file["sound_pressure"]

# Convert to DataFrame
df = pd.DataFrame(data)
print(df)
