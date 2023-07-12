import h5py
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def main(input_dir, output_dir):
    OUT_FILE_SIZE = 50000

    exam_ids_buffer = []
    tracings_buffer = []
    K = 0

    os.mkdir(output_dir)
    df = pd.read_csv(f'{input_dir}/exams.csv')
    df.set_index('exam_id', inplace=True)
    to_remove_indices = []

    for fn in tqdm(os.listdir(input_dir), desc="Processing files"):
        path = os.path.join(input_dir, fn)

        if not path.endswith('.hdf5'):
            continue

        failed_count = 0

        with h5py.File(path) as f:
            tracings = f['tracings'][:-1]
            transformed_tracings = np.zeros((len(tracings), 4096, 8), dtype='i2')
            transformed_tracings[:, :, 0:6] = tracings[:, :, 6:12] / 0.00488
            transformed_tracings[:, :, 6:8] = tracings[:, :, 0:2] / 0.00488

            for i, (tracing, exam_id) in enumerate(zip(transformed_tracings, f['exam_id'][:-1])):
                if np.sum(tracing) == 0:
                    failed_count += 1
                    to_remove_indices.append(exam_id)
                else:
                    tracings_buffer.append(tracing)
                    exam_ids_buffer.append(exam_id)

        if len(exam_ids_buffer) >= OUT_FILE_SIZE:
            assert len(exam_ids_buffer) == len(tracings_buffer)

            to_store_exam_ids = exam_ids_buffer[:OUT_FILE_SIZE]
            to_store_tracings = tracings_buffer[:OUT_FILE_SIZE]

            exam_ids_buffer = exam_ids_buffer[OUT_FILE_SIZE:]
            tracings_buffer = tracings_buffer[OUT_FILE_SIZE:]

            with h5py.File(f'{output_dir}/exams_part_{K}.hdf5', 'w') as out:
                out.create_dataset('exam_id', shape=(len(to_store_exam_ids), ), dtype='i4', data=to_store_exam_ids)
                out.create_dataset('tracings', shape=(len(to_store_tracings), 4096, 8), dtype='i2', data=to_store_tracings,
                                   chunks=(1, 4096, 8))

            K += 1

        print(f"{failed_count=}")

    if len(exam_ids_buffer) > 0:
        with h5py.File(f'{output_dir}/exams_part_{K}.hdf5', 'w') as out:
            out.create_dataset('exam_id', shape=(len(exam_ids_buffer), ), dtype='i4', data=exam_ids_buffer)
            out.create_dataset('tracings', shape=(len(tracings_buffer), 4096, 8), dtype='i2', data=tracings_buffer)

    df = df.drop(to_remove_indices)
    df.to_csv(f'{output_dir}/exams.csv')
    print("Done.")


if __name__ == '__main__':
    main('code15ours', 'code15')


# from h5_pt_dataloader import MultiFileHDF5ECGHandle, ECGDataModule
#
# dm = ECGDataModule('code15ours', 8)
# ds = dm.val_dataloader()
# batch = next(iter(ds))
