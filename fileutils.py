"""
Utilities for file operations
"""
import os

def get_all_files_recursive(folder):
    """
    Docstring for get_all_files_recursive
    
    :param folder: Folder location
    """
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files
