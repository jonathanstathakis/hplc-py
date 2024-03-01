import pandas as pd
from hplc_py.common import prepare_dataset_for_input
def get_asschrom_dset():
    path = "/tests/test_data/test_assessment_chrom.csv"
    dset = pd.read_csv(path)

    cleaned_dset = prepare_dataset_for_input(dset, "x", "y")

    return cleaned_dset
