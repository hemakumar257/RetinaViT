import hashlib
import os
import argparse
from pathlib import Path

def get_file_sha256(file_path):
    """Computes SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_checksums(data_dir, output_file):
    """Generates checksums for all files in a directory and saves them to a file."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist. Skipping.")
        return

    with open(output_file, "w") as f:
        for file in data_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(data_path)
                checksum = get_file_sha256(file)
                f.write(f"{checksum}  {relative_path}\n")
    print(f"Checksums saved to {output_file}")

def verify_checksums(data_dir, checksum_file):
    """Verifies files in a directory against a checksum file."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist. Verification failed.")
        return False

    if not os.path.exists(checksum_file):
        print(f"Checksum file {checksum_file} not found.")
        return False

    mismatches = []
    missing_files = []
    
    with open(checksum_file, "r") as f:
        for line in f:
            checksum, relative_path = line.strip().split("  ", 1)
            file_path = data_path / relative_path
            
            if not file_path.exists():
                missing_files.append(str(relative_path))
            else:
                current_checksum = get_file_sha256(file_path)
                if current_checksum != checksum:
                    mismatches.append(str(relative_path))

    if not mismatches and not missing_files:
        print(f"All files in {data_dir} verified successfully.")
        return True
    else:
        if mismatches:
            print(f"Mismatches found in {data_dir}: {mismatches}")
        if missing_files:
            print(f"Missing files in {data_dir}: {missing_files}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or verify dataset checksums.")
    parser.add_argument("--mode", choices=["generate", "verify"], required=True, help="Mode: generate or verify.")
    parser.add_argument("--dataset", choices=["aptos2019", "messidor2", "idrid"], required=True, help="Target dataset.")
    args = parser.parse_args()

    data_dir = f"datasets/raw/{args.dataset}"
    checksum_file = f"datasets/raw/{args.dataset}_checksums.sha256"

    if args.mode == "generate":
        generate_checksums(data_dir, checksum_file)
    else:
        verify_checksums(data_dir, checksum_file)
