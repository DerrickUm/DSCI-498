
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def compute_indices(data):
    """
    Compute ARVI, ABEI, AWEI_nsh, AWEI_sh and a 3-class surface classification
    using only the Sentinel bands ['B01','B02','B03','B04','B08','B11','B12'].
    
    Returns:
      data (pd.DataFrame): with new columns
      idx_dict (dict): arrays of each computed index
    """
    idx_dict = {}
    band_cols = ['B01','B02','B03','B04','B08','B11','B12']
    
    # Only proceed if all required bands exist
    if all(b in data.columns for b in band_cols):
        # 1) calculate indices
        arvi     = (data['B08'] - 2*data['B04'] + data['B02']) / (data['B08'] + 2*data['B04'] + data['B02'])
        abei     = (0.312*data['B01'] + 0.513*data['B02']
                   -0.086*data['B03'] -0.441*data['B04']
                   +0.052*data['B08'] -0.198*data['B11']
                   +0.278*data['B12'])
        awei_nsh = 4*(data['B03'] - data['B11']) - (0.25*data['B08'] + 2.75*data['B12'])
        awei_sh  = (data['B02'] + 2.5*data['B03']
                   -1.5*(data['B08'] + data['B11'])
                   -0.25*data['B12'])

        # 2) clean infinities
        for arr in (arvi, abei, awei_nsh, awei_sh):
            arr.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 3) store in DataFrame
        data['ARVI']     = arvi
        data['ABEI']     = abei
        data['AWEI_nsh'] = awei_nsh
        data['AWEI_sh']  = awei_sh
        data['classification'] = classify_surface(awei_nsh, awei_sh)

        # 4) populate idx_dict
        idx_dict.update({
            'ARVI':     arvi.values,
            'ABEI':     abei.values,
            'AWEI_nsh': awei_nsh.values,
            'AWEI_sh':  awei_sh.values
        })
    else:
        # if missing bands, fill with NaN
        for name in ['ARVI','ABEI','AWEI_nsh','AWEI_sh','classification']:
            data[name] = np.nan

    return data, idx_dict


def classify_surface(awei_nsh, awei_sh):
    """
    Classify surfaces using a decision rule based on AWEI indices.
    Returns an array of classification values:
      - 1 for high albedo (if AWEI_nsh > 0.5)
      - 2 for shadow/dark surfaces (if AWEI_nsh <= 0.5 and AWEI_sh > 0.5)
      - 3 for other surfaces (if AWEI_nsh <= 0.5 and AWEI_sh <= 0.5)
    """
    classification = np.zeros_like(awei_nsh, dtype=int)
    classification[awei_nsh > 0.5] = 1
    classification[(awei_nsh <= 0.5) & (awei_sh > 0.5)] = 2
    classification[(awei_nsh <= 0.5) & (awei_sh <= 0.5)] = 3
    return classification


def classify_water_two_step(data, thresh_nsh=0.0, thresh_sh=0.0, col_name="is_water"):
    """
    Two-step water classification for urban backgrounds where both bright (high-albedo)
    and shadow/dark surfaces can confuse water detection.
    
    Step 1: Mark potential water where AWEI_nsh >= thresh_nsh.
    Step 2: Among those, keep only if AWEI_sh >= thresh_sh.
    
    Stores binary classification in data[col_name] (1 for water, 0 for non-water).
    """
    if "AWEI_nsh" not in data.columns or "AWEI_sh" not in data.columns:
        raise ValueError("AWEI_nsh and AWEI_sh not found. Run compute_indices first.")
    
    data[col_name] = 0
    pre_water_mask = (data["AWEI_nsh"] >= thresh_nsh)
    final_water_mask = pre_water_mask & (data["AWEI_sh"] >= thresh_sh)
    data.loc[final_water_mask, col_name] = 1
    return data

# Example usage:
# Assuming `df` is your DataFrame containing the appropriate bands (both Bxx and Landsat names),
# you can compute indices as follows:
#
# df_updated, indices = compute_indices(df)
# print(df_updated.head())
# print(indices.keys())
