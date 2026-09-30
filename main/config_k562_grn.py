"""Config template for K562 GRN reconstruction.
"""

MODE = "grn"

SCENARIO = "k562_grn"
ADATA_PATH = "Data/mode1/k562/k562.h5ad"
CELLORDER_FILE = "Data/mode1/k562/cell_order.json"
OUTPUT_DIR = "reproducibility_outputs/k562_grn"

# Non-Beeline scenarios automatically use the general GRN profile. K562 keeps
# the signed CDF prior used by the original reconstruction experiment.
GRN_CDF_MODE = "sign"

# Optional GRN evaluation. For a row-subset .npy ground truth, the JSON list
# defines the TF name and row order represented by that matrix.
SELECTED_TF_JSON = None
GROUND_TRUTH_PATH = None

# Optimization / runtime.
MAX_ITER = 10000
K = 1
POOL_SIZE = 30
DOWN_SAMPLE_SIZE = 0
SAVE_EVERY = 100
FLUSH_EVERY = 10
RANDOM_STATE = 0
