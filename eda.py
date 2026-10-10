import numpy as np
import pandas as pd
import data

clinical = pd.read_csv("data.CFB-GBM_clinical_data_v03_20260619.tsv", sep='\t')
treatment = pd.read_csv("data.CFB-GBM_treatment_data_v02_20260129.tsv", sep='\t')
rano = pd.read_csv("data.CFB-GBM_rano_criteria_v04_20260909.tsv", sep='\t')
features = pd.read_csv("data.CFB_GBM_features_extraction_pyradiomics_v03_20260619.tsv", sep='\t')