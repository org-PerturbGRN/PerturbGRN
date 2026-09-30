"""Example configuration for GRN reconstruction with a user dataset.

The paths below are placeholders. Prepare the AnnData and cell-order files as
described in README.md, update the paths, and copy this file to main/config.py.
"""

MODE = "grn"

# Experiment name and repository-relative paths.
SCENARIO = "my_mode1_grn"
ADATA_PATH = "Data/my_mode1/perturbseq.h5ad"
CELLORDER_FILE = "Data/my_mode1/cell_order.json"
OUTPUT_DIR = "outputs/my_mode1_grn"

# The general default is "abs". Use "sign" only when the signed perturbation
# direction is intentionally part of the experiment, as in the K562 template.
GRN_CDF_MODE = "abs"

# Optional evaluation files. SELECTED_TF_JSON is a JSON list of TF symbols
# defining an evaluation row subset and requires GROUND_TRUTH_PATH.
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
