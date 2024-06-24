import os
import shutil

def test_import_overeasy(capsys):
    # Move ~/.overeasy to ~/.overeasy_backup
    home_dir = os.path.expanduser("~")
    original_path = os.path.join(home_dir, ".overeasy")
    backup_path = os.path.join(home_dir, ".overeasy_backup")
    if os.path.exists(original_path):
        shutil.move(original_path, backup_path)
    
    try:
        with capsys.disabled():
            import overeasy
        captured = capsys.readouterr()
        assert captured.out == ""
    finally:
        # Move ~/.overeasy_backup back to ~/.overeasy
        if os.path.exists(backup_path):
            shutil.move(backup_path, original_path)