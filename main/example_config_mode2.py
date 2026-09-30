"""Example configuration for driver inference with a user dataset.

The paths below are placeholders. The AnnData and cell-order JSON must already
follow the prepared Mode 2 schema described in README.md.
"""

MODE = "driver"

# Experiment name and repository-relative paths.
SCENARIO = "my_mode2_driver"
ADATA_PATH = "Data/my_mode2/perturbgrn_input.h5ad"
CELLORDER_FILE = "Data/my_mode2/cell_order.json"
OUTPUT_DIR = "outputs/my_mode2_driver"

# Label of the target-state trajectory in both adata.obs["pert"] and the
# cell-order JSON. Replace this when the prepared data use another label.
TARGET_PERT_LABEL = "unknown"

# Stop once the network gain has stabilized within the configured window.
DRIVER_EARLY_STOP = True
DRIVER_STOP_THRESHOLD = 0.005
DRIVER_STOP_WINDOW = 3

# Ground-truth GRN evaluation is available only in Mode 1.
SELECTED_TF_JSON = None
GROUND_TRUTH_PATH = None

# Runtime settings normally exposed to users.
MAX_ITER = 10000
K = 1
POOL_SIZE = 30
DOWN_SAMPLE_SIZE = 0
SAVE_EVERY = 100
FLUSH_EVERY = 10
RANDOM_STATE = 0
