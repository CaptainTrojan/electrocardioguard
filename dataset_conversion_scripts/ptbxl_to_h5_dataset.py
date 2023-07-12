import h5py
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import wfdb
from collections import Counter


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(f'{input_dir}/ptbxl_database.csv')
    df.rename(columns={'ecg_id': 'exam_id', 'sex': 'is_male'}, inplace=True)
    df['is_male'] = df['is_male'].replace({0: 1, 1: 0})
    df['patient_id'] = df['patient_id'].astype(int)
    df.set_index('exam_id', inplace=True)

    exam_ids_buffer = []
    tracings_buffer = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        data = wfdb.rdsamp(f'{input_dir}/' + row['filename_hr'])
        array = data[0]
        new_array = np.zeros((4096, 12), dtype=np.short)

        assert array.shape[0] == 5000
        clip_size = (5000 - 4096) // 2
        clipped = array[clip_size:-clip_size]
        new_array[:, 0:6] = clipped[:, 6:12] / 0.00488
        new_array[:, 6:8] = clipped[:, 0:2] / 0.00488
        tracings_buffer.append(new_array)
        exam_ids_buffer.append(idx)

    with h5py.File(f'{output_dir}/exams_part_0.hdf5', 'w') as f:
        f.create_dataset('exam_id', data=exam_ids_buffer, dtype='i4')
        f.create_dataset('tracings', data=tracings_buffer, chunks=(1, 4096, 8), dtype='i2')

    df.to_csv(f'{output_dir}/exams.csv')
    print('done')


if __name__ == '__main__':
    main('ptbxl', 'ptbxlours')


# for fn in tqdm(os.listdir('code15'), desc="Processing files"):
#     path = os.path.join('code15', fn)
#
#     if not path.endswith('.hdf5'):
#         continue
#
#     failed_count = 0
#
#     with h5py.File(path) as f:
#         tracings = f['tracings'][:-1]
#         transformed_tracings = np.zeros((len(tracings), 4096, 8), dtype='i2')
#         transformed_tracings[:, :, 0:6] = tracings[:, :, 6:12] / 0.00488
#         transformed_tracings[:, :, 6:8] = tracings[:, :, 0:2] / 0.00488
#
#         for i, (tracing, exam_id) in enumerate(zip(transformed_tracings, f['exam_id'][:-1])):
#             if np.sum(tracing) == 0:
#                 failed_count += 1
#                 to_remove_indices.append(exam_id)
#             else:
#                 tracings_buffer.append(tracing)
#                 exam_ids_buffer.append(exam_id)
#
#     if len(exam_ids_buffer) >= OUT_FILE_SIZE:
#         assert len(exam_ids_buffer) == len(tracings_buffer)
#
#         to_store_exam_ids = exam_ids_buffer[:OUT_FILE_SIZE]
#         to_store_tracings = tracings_buffer[:OUT_FILE_SIZE]
#
#         exam_ids_buffer = exam_ids_buffer[OUT_FILE_SIZE:]
#         tracings_buffer = tracings_buffer[OUT_FILE_SIZE:]
#
#         with h5py.File(f'code15ours/exams_part_{K}.hdf5', 'w') as out:
#             out.create_dataset('exam_id', shape=(len(to_store_exam_ids), ), dtype='i4', data=to_store_exam_ids)
#             out.create_dataset('tracings', shape=(len(to_store_tracings), 4096, 8), dtype='i2', data=to_store_tracings,
#                                chunks=(1, 4096, 8))
#
#         K += 1
#
#     print(f"{failed_count=}")
#
# if len(exam_ids_buffer) > 0:
#     with h5py.File(f'code15ours/exams_part_{K}.hdf5', 'w') as out:
#         out.create_dataset('exam_id', shape=(len(exam_ids_buffer), ), dtype='i4', data=exam_ids_buffer)
#         out.create_dataset('tracings', shape=(len(tracings_buffer), 4096, 8), dtype='i2', data=tracings_buffer)
#
# df = df.drop(to_remove_indices)
# df.to_csv('code15ours/exams.csv')
# print("Done.")
#
#
# # from h5_pt_dataloader import MultiFileHDF5ECGHandle, ECGDataModule
# #
# # dm = ECGDataModule('code15ours', 8)
# # ds = dm.val_dataloader()
# # batch = next(iter(ds))
