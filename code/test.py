import sys
import os

from main import set_seed, setup_logging, train_model, test_model, evaluate_model


def train_test_eval(cfg: str, seed: int = 0):
    set_seed(seed)
    setup_logging()

    args_train = {"train": True, "training_cfg": cfg}
    results_path = train_model(args_train)

    args_test = {"test": True, "eval_path": results_path}
    test_model(args_test)

    args_eval = {"evaluate": True, "eval_path": results_path}
    evaluate_model(args_eval)


def main():
    if len(sys.argv) > 1:
        tests_path = sys.argv[1]
    else:
        print("Usage: python test.py <tests_path>")
        exit(0)

    full_tests_path = os.path.join("../configs/training", tests_path)
    if not os.path.exists(full_tests_path):
        print(f"Error: {full_tests_path} does not exist")
        exit(1)

    line = "-" * 80
    tests = os.listdir(full_tests_path)

    for test in tests:
        test = test.replace(".yaml", "")
        test_path = os.path.join(tests_path, test)

        print(f"{line}\nTesting {test_path}\n{line}\n")
        train_test_eval(test_path)


if __name__ == "__main__":
    main()
