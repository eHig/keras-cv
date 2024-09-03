import urllib.request

import kagglehub

from keras_cv.src.utils import preset_utils


def download_public_file(url, destination_file_name):
    """Downloads a file from a public URL."""
    urllib.request.urlretrieve(url, destination_file_name)
    print(f"File downloaded to {destination_file_name}.")

# Example usage
bucket = "keras-cv-kaggle"
path = "efficientnetv1_b0"
url = f"https://storage.googleapis.com/{bucket}/{path}"
destination_file_name = 'foo'

if __name__ == '__main__':
    # download_public_file(url, destination_file_name)
    kaggle_handle = "gs://keras-cv-kaggle/efficientnetv1_b0"
    # kagglehub.model_download(kaggle_handle, path)

    a = preset_utils.get_file(kaggle_handle, "config.json")
    print(a)
