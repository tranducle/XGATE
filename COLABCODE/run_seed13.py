import sys
import importlib
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python run_seed13.py <experiment_module> [args...]")
    print("Example: python run_seed13.py run_ablation_study --results-root ...")
    sys.exit(1)

module_name = sys.argv[1].replace(".py", "")
module = importlib.import_module(module_name)

# Override multi-run properties
module.N_RUNS = 1
module.SEEDS = [13]

# Remove the wrapper from sys.argv so the target module parses its own args correctly
sys.argv = [sys.argv[0]] + sys.argv[2:]

if __name__ == "__main__":
    module.main()
