# Read file and parse values
import os
import numpy as np

def findmin(filename):
    rows = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(current_dir + filename, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            # Remove leading/trailing whitespace and convert to float where needed
            parts = [p.strip() for p in parts]
            rows.append(parts)

    # Find index of row with minimum 5th column (index 4)
    min_index = min(range(len(rows)), key=lambda i: float(rows[i][4]))

    # Get the full row and convert to LaTeX-friendly string
    min_row = rows[min_index]
    latex_row = ', '.join(min_row)

    # Output
    print(f"Row: {min_index + 1}")  # LaTeX rows are 1-indexed
    print(f"LaTeX-formatted row: {latex_row}")

def finddev(filename):
    data = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(current_dir + filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Skip the first column (timestamp)
            numeric_values = [float(x.strip()) for x in parts[1:]]
            data.append(numeric_values)

    # Convert to numpy array for easy stats
    arr = np.array(data)

    # Calculate mean and std deviation column-wise
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    # Print the results
    for i, (m, s) in enumerate(zip(mean, std), start=1):
        print(f"Column {i}: mean = {m:.6f}, std = {s:.6f}")

findmin("./gridsearch.txt")
finddev("./test_all.txt")

