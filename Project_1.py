import os                       # file paths 
import pandas as pd             # load and work with dataset table 
import numpy as np              # math ops 
import matplotlib.pyplot as plt # graphs 

#----------- FILE PATH 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "rul_hrs.csv")

#----------- DATASET
# read the file into pandas 
df = pd.read_csv(CSV_PATH)
df = df.iloc[:10000].copy()

if "rul" not in df.columns:
    raise ValueError("Dataset must contain a 'rul' column.")

# find and sort all sensor columns 
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
sensor_cols = sorted(sensor_cols)

print("Rows used: ", len(df))
print("Number of sensor columns:", len(sensor_cols))

# keep only sensor cols + rul 
use_cols = sensor_cols + ["rul"]
df = df[use_cols].copy()

# convert values to numeric in case values are stored as text 
for col in use_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# fill missing values if any 
df = df.ffill().bfill()

#----------- RUL CATEGORIES 
Q10 = df["rul"].quantile(0.10)
Q40 = df["rul"].quantile(0.40)
Q90 = df["rul"].quantile(0.90)

# 4 classes 
def get_class(r):
    if r < Q10:
        return "Extremely Low RUL"
    elif r >= Q10 and r < Q40:
        return "Moderately Low RUL"
    elif r >= Q40 and r < Q90:
        return "Moderately High RUL"
    else: 
        return "Extremely High RUL"
    
df["rul_class"] = df["rul"].apply(get_class)
print("\nRUL quantiles:")
print("Q10 =", Q10)
print("Q40 =", Q40)
print("Q90 =", Q90)

#----------- Task 1 — Divide‑and‑Conquer Segmentation
print("\n================ TASK 1 ================")
# step 1: select 10 random sensors 
np.random.seed(13)
selected_sensors = np.random.choice(sensor_cols, 10, replace=False)
print("\nSelected sensors for Task 1:")
print(selected_sensors)

# step 2: recursive segmentation 
def recursive_segm(signal, start, end, threshold, min_size=50):
    segment = signal[start:end]
    variance = np.var(segment)
    # stop splitting if segment small enough 
    if len(segment) <= min_size or variance <= threshold:
        return [(start, end)]
    # divide into 2 halves 
    mid = (start + end) // 2
    left = recursive_segm(signal, start, mid, threshold, min_size)
    right = recursive_segm(signal, mid, end, threshold, min_size)
    return left + right 

# step 3: visualization 
def plot_segm(signal, segments, sensor_name):
    plt.figure(figsize=(12,4))
    plt.plot(signal, label=sensor_name, linewidth = 1)

    # draw a vertical line at the start of each segment
    for (start, end) in segments:
        plt.axvline(start, linestyle="--", linewidth=0.7)

    # draw the last boundary at the end of the signal
    plt.axvline(len(signal) - 1, linestyle="--", linewidth=0.7)

    plt.title(f"Segmentation for {sensor_name}")
    plt.xlabel("Time Index")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

task1_results = []
# loop through the 10 random sensors
for sensor in selected_sensors:
    # get the signal values for the current sensor
    signal = df[sensor].values
      
    # threshold: half of the variance of the full signal
    threshold = 0.5 * np.var(signal)

    # run recursion
    segments = recursive_segm(signal, 0, len(signal), threshold, min_size=50)

    # step 4: segmenation complexity score 
    complexity_score = len(segments)

    # for each final segment find the class
    segment_info = []
    for start, end in segments:
        segment_classes = df["rul_class"].iloc[start:end]
        dominant_class = segment_classes.mode().iloc[0]
        segment_info.append((start, end, dominant_class))

    # save results for this sensor
    task1_results.append({
        "sensor": sensor,
        "complexity_score": complexity_score,
        "segments": segment_info})
    
    # print a short summary
    print(f"\nSensor: {sensor}")
    print("Segmentation Complexity Score:", complexity_score)
    print("First 5 segments:", segment_info[:5])

    # visualize segmentation
    plot_segm(signal, segments, sensor)

# summary table for Task 1
task1_summary = pd.DataFrame([
    {
        "sensor": item["sensor"],
        "segmentation_complexity_score": item["complexity_score"]
    }
    for item in task1_results
])

print("\nTask 1 Summary (Segmentation Complexity Score):")
print("-" * 40)
print(f"{'Sensor':<12} {'Complexity Score':<20}")
print("-" * 40)

for i, row in task1_summary.iterrows():
    print(f"{row['sensor']:<12} {row['segmentation_complexity_score']:<20}")


#----------- Task 2 — Divide‑and‑Conquer Clustering of Segments
print("\n================ TASK 2 ================")

# step 1: clustering 
# each row = one time instance
# each column = one sensor measurement
X = df[sensor_cols].values.astype(float)

# normalize the data 
means = X.mean(axis=0)
stds = X.std(axis=0)

# avoid division by zero if a sensor has no variation
stds[stds == 0] = 1
X = (X - means) / stds

# helper function: sum of squared errors sse 
def sse(points):
    if len(points) == 0:
        return 0
    center = np.mean(points, axis=0)
    return np.sum((points - center) ** 2)

# helper function: split one cluster into two 
def split(points, indexes):
    if len(points)<=1:
        return[(points, indexes)]

    # find variance of each feature inside the cluster
    variances = np.var(points, axis=0)

    # choose the feature with the largest variance
    split_feature = np.argmax(variances)

    # split at the median 
    median_value = np.median(points[:, split_feature])

    # left cluster
    left_cluster = points[:, split_feature] <= median_value

    # right cluster
    right_cluster = points[:, split_feature] > median_value

    left_points = points[left_cluster]
    right_points = points[right_cluster]

    left_indexes = indexes[left_cluster]
    right_indexes = indexes[right_cluster]

    # if one side becomes empty, split the cluster in half by position
    if len(left_points) == 0 or len(right_points) == 0:
        half = len(points) // 2
        left_points = points[:half]
        right_points = points[half:]
        left_indexes = indexes[:half]
        right_indexes = indexes[half:]

    return [(left_points, left_indexes), (right_points, right_indexes)]

# top-down clustering 
def top_down_clust(X, k=4):
    clusters = [(X, np.arange(len(X)))]
    while len(clusters) < k:
        cluster_sse = [sse(points) for points, indx in clusters]
        split_idx = np.argmax(cluster_sse)
        points, indexes = clusters.pop(split_idx)
        new_clusters = split(points, indexes)
        clusters.extend(new_clusters)
    return clusters 

# create 4 clusters 
clusters = top_down_clust(X, k=4)

# create an array to store the cluster label for each row
cluster_labels = np.zeros(len(df), dtype=int)

# assign each row to its cluster ID
for cluster_id, (points, indexes) in enumerate(clusters):
    cluster_labels[indexes] = cluster_id

# add the cluster labels to the DataFrame
df["cluster_id"] = cluster_labels


# step 2: majority class count per cluster 

task2_results = []

for cluster_id in range(4):
    # get all rows that belong to this cluster
    cluster_data = df[df["cluster_id"] == cluster_id]

    # count how many rows fall into each RUL category
    class_counts = cluster_data["rul_class"].value_counts()

    # the majority class is the dominant class 
    dominant_class = class_counts.idxmax()

    # majority class count = how many times that class appears
    majority_count = class_counts.max()

    # results
    task2_results.append({
        "cluster_id": cluster_id,
        "cluster_size": len(cluster_data),
        "majority_class": dominant_class,
        "majority_count": majority_count
    })

# summary table
task2_summary = pd.DataFrame(task2_results)

print("\nTask 2 Summary (Majority Class Count per Cluster):")
print("-" * 80)
print(f"{'Cluster':<10} {'Size':<10} {'Majority Class':<25} {'Majority Count':<15}")
print("-" * 80)

for i, row in task2_summary.iterrows():
    print(
        f"{row['cluster_id']:<10} "
        f"{row['cluster_size']:<10} "
        f"{row['majority_class']:<25} "
        f"{row['majority_count']:<15}"
    )

#----------- Task 3 — Task 3 — Maximum Subarray (Kadane)
print("\n================ TASK 3 ================")

def kadane(arr): 
    max_sum = arr[0]
    current_sum = arr[0]

    # track best subarray indexes 
    start_indx = 0
    end_indx = 0
    temp_start_indx = 0 

    for i in range(1, len(arr)):
        if arr[i] > current_sum + arr[i]:
            current_sum = arr[i]
            temp_start_indx = i
        else:
            current_sum += arr[i]

        # update the result 
        if current_sum > max_sum:
            max_sum = current_sum
            start_indx = temp_start_indx
            end_indx = i
    return max_sum, start_indx, end_indx

# list to store Task 3 results
task3_results = []

# step 4: repeat for all available sensors
for sensor in sensor_cols:
    # get the sensor signal
    signal = df[sensor].values

    # step 1: Preprocess a Sensor Signal

    #absolute first difference 
    d = np.abs(np.diff(signal))

    # subtract the mean of the absolute differences
    x = d - np.mean(d)

    # step 2: Apply Kadane’s Algorithm

    max_sum, start_idx, end_idx = kadane(x)

    interval_start = start_idx
    interval_end = end_idx + 1

    # step 3: Compare With RUL Categories

    # get the RUL classes
    interval_classes = df["rul_class"].iloc[interval_start:interval_end + 1]

    # find the dominant RUL class in that interval
    dominant_class = interval_classes.mode().iloc[0]

    # store the result
    task3_results.append({
        "sensor": sensor,
        "max_deviation_sum": max_sum,
        "start_index": interval_start,
        "end_index": interval_end,
        "dominant_rul_class": dominant_class
    })


# summary table
task3_summary = pd.DataFrame(task3_results)

# sort by highest maximum deviation sum
task3_summary = task3_summary.sort_values(
    by="max_deviation_sum",
    ascending=False
).reset_index(drop=True)

print("\nTask 3 Summary (Kadane Maximum Deviation Intervals):")
print("-" * 110)
print(f"{'Sensor':<12} {'Max Deviation':<18} {'Start':<10} {'End':<10} {'Dominant RUL Class':<25}")
print("-" * 110)

for i, row in task3_summary.iterrows():
    print(
        f"{row['sensor']:<12} "
        f"{row['max_deviation_sum']:<18.4f} "
        f"{row['start_index']:<10} "
        f"{row['end_index']:<10} "
        f"{row['dominant_rul_class']:<25}"
    )




