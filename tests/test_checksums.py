import os
import pytest
import shutil
from pathlib import Path
from retinavit.data.checksums import generate_checksums, verify_checksums

@pytest.fixture
def temp_data_dir(tmp_path):
    """Creates a temporary directory with dummy files."""
    d = tmp_path / "dummy_dataset"
    d.mkdir()
    (d / "file1.txt").write_text("hello world")
    (d / "subdir").mkdir()
    (d / "subdir" / "file2.txt").write_text("retinavit test")
    return d

def test_checksum_lifecycle(temp_data_dir, tmp_path):
    checksum_file = tmp_path / "checksums.sha256"
    
    # 1. Generate
    generate_checksums(temp_data_dir, checksum_file)
    assert checksum_file.exists()
    
    # 2. Verify (should pass)
    assert verify_checksums(temp_data_dir, checksum_file) is True
    
    # 3. Modify a file (should fail mismatch)
    (temp_data_dir / "file1.txt").write_text("modified content")
    assert verify_checksums(temp_data_dir, checksum_file) is False
    
    # 4. Remove a file (should fail missing)
    (temp_data_dir / "file1.txt").unlink()
    assert verify_checksums(temp_data_dir, checksum_file) is False
