"""
Download the CMS HRRP (Hospital Readmissions Reduction Program) dataset from Kaggle.

Source: https://www.kaggle.com/datasets/jvstinjtvy/cms-hrrp
File: Hospitals_Readmissions_Reduction_Program_ready.csv (~876 KB)

Prerequisites:
    pip install kaggle

    # 1. Go to https://www.kaggle.com/settings/account
    # 2. Click "Create New API Token" -> downloads kaggle.json
    # 3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

Usage:
    python download_data.py
"""
import os
import sys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Please install kaggle CLI: pip install kaggle")
    print("Then set up your API token as described above.")
    sys.exit(1)


def main():
    dataset = "jvstinjtvy/cms-hrrp"
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Downloading {dataset}...")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path=output_dir, unzip=True)

    print(f"Done! File saved to: {output_dir}")
    print("Run:  python spc_dashboard.py  to generate hospital SPC reports.")
    print("Run:  python kpi_comparison.py  to generate the KPI comparison dashboard.")


if __name__ == "__main__":
    main()
