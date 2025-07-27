# Read file and parse values
import os
rows = []
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(current_dir + "./gridsearch.txt", "r") as f:
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