#!/usr/bin/env python3
"""
Script to delete images listed in delete.txt, but skip deletion if it would be the last .jpg file in the folder.
"""

import os
import glob
from pathlib import Path

def count_jpg_files_in_folder(folder_path):
    """Count the number of .jpg files in a given folder."""
    if not os.path.exists(folder_path):
        return 0
    
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    return len(jpg_files)

def delete_images_from_file(delete_file_path="delete.txt"):
    """Delete images listed in delete.txt, skipping if it would be the last .jpg in folder."""
    
    if not os.path.exists(delete_file_path):
        print(f"Error: {delete_file_path} not found!")
        return
    
    deleted_count = 0
    skipped_count = 0
    error_count = 0
    skipped_folders = set()
    
    print(f"Reading deletion list from {delete_file_path}...")
    
    with open(delete_file_path, 'r') as f:
        lines = f.readlines()
    
    total_files = len(lines)
    print(f"Found {total_files} files to process...")
    
    for i, line in enumerate(lines, 1):
        file_path = line.strip()
        
        if not file_path:
            continue
            
        print(f"[{i}/{total_files}] Processing: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"  ⚠️  File not found, skipping: {file_path}")
            error_count += 1
            continue
        
        # Get the folder containing this file
        folder_path = os.path.dirname(file_path)
        
        # Count .jpg files in the folder
        jpg_count = count_jpg_files_in_folder(folder_path)
        
        if jpg_count <= 1:
            print(f"  🚫 SKIPPED: Would be the last .jpg file in folder '{folder_path}' (current count: {jpg_count})")
            skipped_count += 1
            skipped_folders.add(folder_path)
            continue
        
        # Safe to delete - not the last .jpg file
        try:
            os.remove(file_path)
            print(f"  ✅ Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ Error deleting {file_path}: {e}")
            error_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("DELETION SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_files}")
    print(f"Successfully deleted: {deleted_count}")
    print(f"Skipped (last .jpg in folder): {skipped_count}")
    print(f"Errors: {error_count}")
    
    if skipped_folders:
        print(f"\nFolders with skipped files (would be left with no .jpg files):")
        for folder in sorted(skipped_folders):
            print(f"  - {folder}")
    
    print(f"\nRemaining files in delete.txt: {total_files - deleted_count - error_count}")

if __name__ == "__main__":
    delete_images_from_file()
