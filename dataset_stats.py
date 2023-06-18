import os

from h5_pt_dataloader import MultiFileHDF5ECGHandle

for ds in os.listdir('datasets'):
    full_path = os.path.join('datasets', ds)
    h = MultiFileHDF5ECGHandle(full_path)
    count_signals = len(h)
    count_patients = h.metadata_df['patient_id'].nunique()
    print(full_path, count_signals, count_patients)
