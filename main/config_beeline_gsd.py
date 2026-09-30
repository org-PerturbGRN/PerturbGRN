"""Config template for reproducing the Beeline GSD GRN reconstruction.
"""

MODE = "grn"

SCENARIO = "GSD_beeline_grn"
ADATA_PATH = "Data/mode1/GSD/GSD.h5ad"
CELLORDER_FILE = "Data/mode1/GSD/cell_order.json"
OUTPUT_DIR = "reproducibility_outputs/mode1_gsd"

# The GSD scenario name automatically selects the legacy Beeline profile.

# Optional GRN evaluation. SELECTED_TF_JSON restricts metrics to named
# regulator rows and is used only together with GROUND_TRUTH_PATH.
SELECTED_TF_JSON = None
GROUND_TRUTH_PATH = "Data/mode1/GSD/refNetwork.csv"

# Optimization / runtime.
MAX_ITER = 341
K = 1
POOL_SIZE = 1
DOWN_SAMPLE_SIZE = 0
SAVE_EVERY = 100
FLUSH_EVERY = 10
RANDOM_STATE = 0
