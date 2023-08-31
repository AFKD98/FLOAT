import pickle
import numpy as np

#read a pickle file
def read_pickle(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data

#read a text file and store all numbers in an array
def read_txt(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    return np.array([float(line.strip()) for line in lines])

device_trace = "client_device_capacity"
compute_trace_specific_model = "compute_trace_efficient_net_b4.txt"
device_trace_new = "client_device_capacity_efficient_net_b4"

model_compute = read_txt(compute_trace_specific_model)
#reverse the model_compute list
model_compute = model_compute[::-1]
data = read_pickle(device_trace)
# Convert dictionary items to a list of key-value pairs
items = list(data.items())

i = 0
# Print the first three items
for key, value in data.items():
    #replace the value with the compute of the specific model in reverse order (from the last to the first)
    value['computation'] = model_compute[0]
    #store the value back to the dictionary
    data[key] = value
    i+=1
    if i == len(model_compute):
        i = 0
    # print(f"Key: {key}, Value: {data[key]}")


#save the updated dictionary to a pickle file
with open(device_trace_new, "wb") as f:
    pickle.dump(data, f)
    