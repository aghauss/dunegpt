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
    # Get the absolute path to the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to base_dir
    input_dir = os.path.join(base_dir, '../../data/raw/Dune_texts')
    output_file = os.path.join(base_dir, '../../data/processed/dune_full_corpus.txt')
    
    # Call the function with the new absolute paths
    merge_text_files(input_dir, output_file)

if __name__ == '__main__':
    main()