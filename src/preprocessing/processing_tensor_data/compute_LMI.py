
import os
import sys

import numpy as np
import pandas as pd

from fast_learning import paths, database
from fast_learning import imaging


# =============================================================================
# Compute LMI.
# =============================================================================

# This performs ROC analysis on each cell with mapping trials.
# Mapping trial of Day 0 are not included in the analysis.

# Parameters.
append_results = False
response_win = (0, 0.300)
baseline_win = (-1, 0)
nshuffles = 1000

# Get directories and files.
db_path = paths.db_path
nwb_path = paths.nwb_dir
processed_data_folder = paths.processed_dir
result_file = os.path.join(processed_data_folder, 'lmi_results.csv')

# Get mice list.
days = ['-3', '-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = database.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

# Load results if already computed.
if not os.path.exists(result_file):
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'lmi', 'lmi_p'])
else:
    df_results = pd.read_csv(result_file)
if not append_results:
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'lmi', 'lmi_p'])

df = []
for mouse_id in mice_list:
    if df_results.loc[df_results.mouse_id==mouse_id].shape[0] > 0:
        print(f'Mouse {mouse_id} already done. Skipping.')
        continue
    print(f'Processing {mouse_id}')
    # Load through load_mouse_xarray so artefact cells are excluded as in
    # every other analysis.
    data_mapping = imaging.load_mouse_xarray(
        mouse_id, paths.tensor_dir, 'tensor_xarray_mapping_data.nc', subtracted=False)
    data_mapping = data_mapping - np.nanmean(data_mapping.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    
    data_pre = data_mapping.sel(trial=data_mapping.coords['day'].isin([-2, -1]))
    data_pre = data_pre.sel(time=slice(*response_win)).mean(dim='time')
    data_post = data_mapping.sel(trial=data_mapping.coords['day'].isin([1, 2]))
    data_post = data_post.sel(time=slice(*response_win)).mean(dim='time')

    lmi, lmi_p = imaging.compute_roc(data_pre, data_post, nshuffles=nshuffles)
    df.append(pd.DataFrame({'mouse_id': mouse_id,
                            'roi': data_mapping.roi.values,
                            'cell_type': data_mapping.cell_type.values,
                            'lmi': lmi, 'lmi_p': lmi_p}))
if len(df)>0:
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    df_results = pd.concat([df_results, df])
    os.makedirs(processed_data_folder, exist_ok=True)
    df_results.to_csv(result_file, index=False)
else:
    print('No new data to process.')

