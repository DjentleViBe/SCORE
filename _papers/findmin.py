# Read file and parse values
import os
import numpy as np
import matplotlib.pyplot as plt

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

def readfile(filename):
    data = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(current_dir + filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Skip the first column (timestamp)
            numeric_values = [float(x.strip()) for x in parts[0:]]
            data.append(numeric_values)
    return np.array(data)

def test_cases(args):
    data_collect = []
    for arg in args:
        print(f"Testing with argument: {arg}")
        data_collect.append(readfile(f"/../RESULTS/{arg}/{arg}.csv"))
        # Here you would call the actual test function with arg
        # For example: run_test(arg)
    data_collect = np.array(data_collect)
    median_data = np.median(data_collect, axis=0)
    plt.figure(figsize=(6, 3))
    plt.plot(median_data[:, 0], label='Training Loss')
    plt.plot(median_data[:, 1], label='Cross Entropy Loss')
    plt.plot(median_data[:, 2], label='Repetition Loss')
    plt.plot(median_data[:, 3], label='Sequence Loss')
    plt.plot(median_data[:, 4], label='Validation Loss')
    plt.plot(median_data[:, 5], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc="center left",bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig('./_papers/ICLR/Loss.pdf')

#findmin("./gridsearch.txt")
test_cases(["CB_Large_5"])

