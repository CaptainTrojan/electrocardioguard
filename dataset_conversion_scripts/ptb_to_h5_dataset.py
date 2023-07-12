import re

import h5py
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import wfdb
from collections import Counter


def main(input_dir, output_dir):

    os.makedirs(f'{output_dir}', exist_ok=True)

    exam_ids_buffer = []
    patient_ids_buffer = []
    tracings_buffer = []

    with open(f'{input_dir}/RECORDS', 'r') as f:
        for idx, p in enumerate(f):
            p = p.strip()

            pattern = r"patient(\d+)"
            match = re.search(pattern, p)
            if match:
                patient_id = int(match.group(1))
            else:
                raise ValueError("Patient ID couldn't have been extracted.")

            path = os.path.join(f"{input_dir}", p)
            data = wfdb.rdsamp(path)

            array = data[0]
            new_array = np.zeros((4096, 12), dtype=np.short)

            clip_size = (array.shape[0] - 8192) // 2
            clipped = array[clip_size:-clip_size]
            clipped = clipped[::2, :][:4096, :]
            new_array[:, 0:6] = clipped[:, 6:12] / 0.00488
            new_array[:, 6:8] = clipped[:, 0:2] / 0.00488
            tracings_buffer.append(new_array)
            exam_ids_buffer.append(idx)
            patient_ids_buffer.append(patient_id)


    with h5py.File(f'{output_dir}/exams_part_0.hdf5', 'w') as f:
        f.create_dataset('exam_id', data=exam_ids_buffer, dtype='i4')
        f.create_dataset('tracings', data=tracings_buffer, chunks=(1, 4096, 8), dtype='i2')

    df = pd.DataFrame.from_dict({'exam_id': exam_ids_buffer, 'patient_id': patient_ids_buffer})
    df.set_index('exam_id', inplace=True)
    df.to_csv(f'{output_dir}/exams.csv')
    print('done')


if __name__ == '__main__':
    main('ptb', 'ptbours')