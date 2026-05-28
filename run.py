import argparse
from experiment import run_lightgbm_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_results', type=bool, default=True)
    args    = parser.parse_args()

    run_lightgbm_pipeline(args.save_results)