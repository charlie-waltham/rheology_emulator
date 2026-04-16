import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from main import evaluate_model, set_seed, setup_logging, test_model, train_model


def train_test_eval(cfg: str, seed: int = 0):
    set_seed(seed)

    args_train = {"train": True, "training_cfg": cfg}
    results_path = train_model(args_train)

    args_test = {"test": True, "eval_path": results_path}
    test_model(args_test)

    args_eval = {"evaluate": True, "eval_path": results_path}
    evaluate_model(args_eval)


def main():
    if len(sys.argv) > 2:
        tests_path = sys.argv[1]
        results_dir = sys.argv[2]
    else:
        print("Usage: python test.py <tests_path> <results_dir>")
        exit(0)

    full_tests_path = Path("../configs/training") / tests_path
    if not full_tests_path.exists():
        print(f"Error: {full_tests_path} does not exist")
        exit(1)
    full_results_dir = Path("../results/nn") / results_dir
    if not full_results_dir.exists():
        os.makedirs(full_results_dir)

    plt.style.use("seaborn-v0_8")

    line = "-" * 80
    tests = list(full_tests_path.iterdir())
    print(f"Found {len(tests)} tests in {full_tests_path}\n")

    existing_results = [p.name for p in full_results_dir.iterdir()]
    print(f"Found {len(existing_results)} existing results in {results_dir}\n")

    setup_logging()

    for test in tests:
        test = test.stem
        config_path = Path(tests_path) / test
        result_path = full_results_dir / test

        result = next(result_path.iterdir(), None)

        if result is not None:
            print(f"{line}\nResult {result} already exists. Skipping training/testing.\n{line}\n")
            evaluate_model({"evaluate": True, "eval_path": str(result)})
        else:
            print(f"{line}\nTesting {config_path}\n{line}\n")
            train_test_eval(str(config_path))


if __name__ == "__main__":
    main()
