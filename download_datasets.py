import glob
import os
import os.path
import shutil
import tempfile

import wget
from tqdm import tqdm

import dataset_conversion_scripts as ds


def download_c15p(root):
    link_base = "https://zenodo.org/record/4916206/files/{}?download=1"
    files_to_download = ['exams.csv']
    for i in range(18):
        files_to_download.append(f'exams_part{i}.zip')

    for f in tqdm(files_to_download, desc="Downloading CODE-15%"):
        download_url = link_base.format(f)
        result_path = os.path.join(root, f)
        wget.download(download_url, result_path)
        if f.endswith('zip'):
            shutil.unpack_archive(result_path, root)
            os.remove(result_path)


def download_ptbxl(root):
    zip_file_dist = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    zip_file_local = os.path.join(root, "ptbxl.zip")
    wget.download(zip_file_dist, zip_file_local)
    shutil.unpack_archive(zip_file_local, root)
    os.remove(zip_file_local)
    shutil.move(os.path.join(root, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500"),
                os.path.join(root, "records500"))
    shutil.move(os.path.join(root, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"),
                os.path.join(root, "ptbxl_database.csv"))


def download_ptb(root):
    zip_file_dist = "https://www.physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip"
    zip_file_local = os.path.join(root, "ptbxl.zip")
    wget.download(zip_file_dist, zip_file_local)
    shutil.unpack_archive(zip_file_local, root)
    os.remove(zip_file_local)
    shutil.move(os.path.join(root, "ptb-diagnostic-ecg-database-1.0.0/RECORDS"),
                os.path.join(root, "RECORDS"))
    for file in glob.glob(os.path.join(root, 'ptb-diagnostic-ecg-database-1.0.0', 'patient*')):
        shutil.move(file,
                    root)


def main():
    ROOT = 'datasets'

    os.makedirs(ROOT, exist_ok=True)

    print("Downloading CODE-15%...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_c15p(tmpdirname)
        print("CODE-15% downloaded, processing...")
        ds.process_c15p(tmpdirname, os.path.join(ROOT, 'c15p'))
        print("CODE-15% processed.")

    print("Downloading PTB-XL...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_ptbxl(tmpdirname)
        print("PTB-XL downloaded, processing...")
        ds.process_ptbxl(tmpdirname, os.path.join(ROOT, 'ptbxl'))
        print("PTB-XL processed")

    print("Downloading PTB...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_ptb(tmpdirname)
        print("PTB downloaded, processing...")
        ds.process_ptb(tmpdirname, os.path.join(ROOT, 'ptb'))
        print("PTB processed")


if __name__ == '__main__':
    main()
