from pathlib import Path
import shutil

# === CONFIGURATION ===
# Set these paths to your source and destination directories
SOURCE_ROOT = Path("/home/guru/Worspace_Mahish/LLM_Post_Processing/01_DataFiltering_02/processed")
DEST_ROOT = Path("/home/guru/Worspace_Mahish/LLM_Post_Processing/01_DataFiltering_02/First_Filtered_Data")


def filter_and_copy(source_root: Path, dest_root: Path) -> None:
    """
    Walk through each subdirectory in source_root and copy any paper directory
    that contains non-empty 'infographic' and 'fulltext' subdirectories into dest_root.
    """
    dest_root.mkdir(parents=True, exist_ok=True)

    for paper_dir in source_root.iterdir():
        if not paper_dir.is_dir():
            continue

        infographic_dir = paper_dir / 'infographic'
        fulltext_dir = paper_dir / 'fulltext'

        has_infographic = infographic_dir.exists() and any(infographic_dir.iterdir())
        has_fulltext = fulltext_dir.exists() and any(fulltext_dir.iterdir())

        if has_infographic and has_fulltext:
            target_dir = dest_root / paper_dir.name
            shutil.copytree(paper_dir, target_dir)
            print(f"Copied: {paper_dir} -> {target_dir}")
        else:
            print(f"Skipping: {paper_dir} (infographic or fulltext missing/empty)")


if __name__ == "__main__":
    filter_and_copy(SOURCE_ROOT, DEST_ROOT)
