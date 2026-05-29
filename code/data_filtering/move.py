import os
import shutil

# === CONFIGURATION ===
# Path to the text file containing folder names (one per line)
LIST_FILE = '/home/guru/Worspace_Mahish/LLM_Post_Processing/01_DataFiltering_02/third_pass_names.txt'
# Source directory with all first-pass folders
SOURCE_DIR = '/home/guru/Worspace_Mahish/LLM_Post_Processing/01_DataFiltering_02/Second_Filtered_Data'
# Target directory where selected folders will be copied
TARGET_DIR = '/home/guru/Worspace_Mahish/LLM_Post_Processing/01_DataFiltering_02/Third_Filtered_Data'
# Set to True to overwrite existing folders in the target directory
OVERWRITE = False
# Set to True to delete folders from the source directory after copying
DELETE_SOURCE = True


def load_folder_list(txt_path):
    """
    Read folder names from a text file, one per line, ignoring blank lines.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: List file '{txt_path}' not found.")
        exit(1)


def copy_and_delete_folders(folder_names, src_dir, dst_dir, overwrite=False, delete_source=False):
    """
    Copy specified folders from source to target directory, optionally overwriting,
    and delete from source if delete_source is True.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for name in folder_names:
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)

        if not os.path.isdir(src_path):
            print(f"Skipping '{name}': not found in source directory.")
            continue

        if os.path.exists(dst_path):
            if overwrite:
                shutil.rmtree(dst_path)
                print(f"Overwriting existing folder '{dst_path}'.")
            else:
                print(f"Skipping '{name}': already exists in target directory.")
                continue

        try:
            shutil.copytree(src_path, dst_path)
            print(f"Copied '{name}' to target directory.")

            if delete_source:
                shutil.rmtree(src_path)
                print(f"Deleted '{name}' from source directory.")

        except Exception as e:
            print(f"Failed to copy/delete '{name}': {e}")


def main():
    # Load list of folders from the static text file
    folders = load_folder_list(LIST_FILE)
    # Perform copying and optional deletion
    copy_and_delete_folders(
        folders,
        SOURCE_DIR,
        TARGET_DIR,
        overwrite=OVERWRITE,
        delete_source=DELETE_SOURCE
    )


if __name__ == '__main__':
    main()
