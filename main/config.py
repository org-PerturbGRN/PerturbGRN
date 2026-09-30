"""Config template for latent driver TF identification.

This path updates the latent driver posterior during training and scores
candidate edges by likelihood only. The posterior E-step matches the local
implementation used in the original t2tcl_trans_linshi_scGPT.py script.
"""

MODE = "driver"

SCENARIO = "fib2dc_driver"
ADATA_PATH = "Data/mode2/fib2dc/perturbgrn_input.h5ad"
CELLORDER_FILE = "Data/mode2/fib2dc/cell_order.json"
OUTPUT_DIR = "reproducibility_outputs/mode2_fib2dc"

# The target trajectory keeps the historical perturbation label used by the
# Fib2DC experiment; other prepared datasets commonly use "unknown".
TARGET_PERT_LABEL = "PU.1+IRF8+BATF3"
DRIVER_EARLY_STOP = True
DRIVER_STOP_THRESHOLD = 0.005
DRIVER_STOP_WINDOW = 3

# Optional evaluation inputs.
SELECTED_TF_JSON = None
GROUND_TRUTH_PATH = None

# Optimization / runtime.
MAX_ITER = 10000
K = 1
POOL_SIZE = 50
DOWN_SAMPLE_SIZE = 3
SAVE_EVERY = 100
FLUSH_EVERY = 10
RANDOM_STATE = 0
