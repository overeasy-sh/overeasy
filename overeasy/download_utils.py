import os
import tempfile
import urllib
import progressbar

class URLProgressBar():
    def __init__(self, filename: str):
        print(f"\nDownloading {filename} ...")
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

# It's important to download files this way to prevent files from being corrupted
# if a error occurs while downloading.
def atomic_retrieve_and_rename(url: str, destination_path: str):
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            urllib.request.urlretrieve(url, tmp_file_path, reporthook=URLProgressBar(os.path.basename(destination_path)))
        os.rename(tmp_file_path, destination_path)
    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise e