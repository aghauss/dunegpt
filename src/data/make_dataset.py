import glob
import os

def merge_text_files(input_dir, output_file):
    """
    Merges text files from a specified directory into a single output file.

    Args:
        input_dir (str): Path to the directory containing text files.
        output_file (str): Path to the output file where the merged text will be stored.
    """
    # Use os.path.join to ensure the path is constructed correctly for any OS
    paths = glob.glob(os.path.join(input_dir, "*.txt"))
    
    # Sort the file paths for consistent order (useful in many contexts)
    paths = sorted(paths)
    
    print(f"Merging {len(paths)} files from {input_dir} into {output_file}...")
    
    with open(output_file, "w", encoding='ISO-8859-1') as outfile:
        for path in paths:
            with open(path, "r", encoding='ISO-8859-1') as infile:
                for line in infile:
                    stripped_line = line.strip()
                    if stripped_line:  # Avoid writing empty lines
                        print(stripped_line, file=outfile)
    
    print("Corpus merged successfully.")

def main():
    drive_path = '../../data/raw/Dune_texts'
    dataset_file = "../../data/processed/dune_full_corpus.txt"  # Save in processed directory
    merge_text_files(drive_path, dataset_file)

if __name__ == '__main__':
    main()
