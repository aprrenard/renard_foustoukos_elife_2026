
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
from sklearn.metrics import auc, roc_curve
from sklearn.utils import shuffle

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging 
from src.utils.utils_behavior import *
from src.utils.utils_imaging import compute_roc
from joblib import Parallel, delayed


# =============================================================================
# Compute LMI.
# =============================================================================

# This perfornms ROC analysis on each cell with mapping trials.
# Mapping trial of Day 0 are not included in the analysis.

# Parameters.
append_results = False
response_win = (0, 0.300)
baseline_win = (-1, 0)
nshuffles = 1000

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')
result_file = os.path.join(processed_data_folder, 'lmi_results.csv')

# Get mice list.
days = ['-3', '-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
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
    data_mapping = xr.open_dataarray(os.path.join(io.tensor_dir, mouse_id, 'tensor_xarray_mapping_data.nc'))
    data_mapping = data_mapping - np.nanmean(data_mapping.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    
    data_pre = data_mapping.sel(trial=data_mapping.coords['day'].isin([-2, -1]))
    data_pre = data_pre.sel(time=slice(*response_win)).mean(dim='time')
    data_post = data_mapping.sel(trial=data_mapping.coords['day'].isin([1, 2]))
    data_post = data_post.sel(time=slice(*response_win)).mean(dim='time')

    lmi, lmi_p = utils_imaging.compute_roc(data_pre, data_post, nshuffles=nshuffles)
    df.append(pd.DataFrame({'mouse_id': mouse_id,
                            'roi': data_mapping.roi.values,
                            'cell_type': data_mapping.cell_type.values,
                            'lmi': lmi, 'lmi_p': lmi_p}))
if len(df)>0:
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    df_results = pd.concat([df_results, df])
    df_results.to_csv(result_file)
else:
    print('No new data to process.')

# data_mapping.shape
# np.isnan(data_mapping).sum()
# data_mapping[0].mean('time')


