import os

def get_all_files_recursive(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files