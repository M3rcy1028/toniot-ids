import argparse
from experiment import run_lightgbm_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--feature_selection', type=bool, default=True)
    parser.add_argument('--fs_config', type=str, default="reports/lightgbm_260730/xai/selection/final_selected_features.csv")
    args    = parser.parse_args()

    run_lightgbm_pipeline(args.save_results, args.feature_selection, args.fs_config)