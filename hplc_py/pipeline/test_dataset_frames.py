"""
:2024-02-28 10:02:05 - Due to the still uncertain nature of the pipeline, development within pytest is proving to be an impediment. Thus the test datasdets will be available for import from here as pandas dataframes
"""

import pandas as pd
ASSCHROM_PATH = "tests/test_data/test_assessment_chrom.csv"
ASSCHROM_DF = pd.read_csv(ASSCHROM_PATH)
TIME_ASSCHROM = "x"
AMP_ASSCHROM = "y"