import os
import pandas as pd

# Seeds to check
seeds = [10623, 11611, 42]
base_dir = 'src/sft/results/cd'
expected_folds = {1, 2, 3, 4, 5}

# Final report dictionary
final_report = {}

for seed in seeds:
    root_dir = os.path.join(base_dir, f'seed_{seed}')
    seed_report = {
        'status': '✅ OK',
        'num_csv_files': 0,
        'missing_folds_files': [],
        'missing_fold_column_files': [],
        'read_errors': [],
        'files_all_folds_ok': []
    }

    if not os.path.isdir(root_dir):
        seed_report['status'] = '❌ Missing directory'
        final_report[seed] = seed_report
        continue

    # Get CSV files only
    files = [f for f in os.listdir(root_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(root_dir, f))]
    seed_report['num_csv_files'] = len(files)

    # Check minimum file count
    if len(files) < 40:
        seed_report['status'] = f'❌ Too few files: {len(files)} (expected ≥ 40)'

    for filename in files:
        filepath = os.path.join(root_dir, filename)
        try:
            df = pd.read_csv(filepath)
            if "fold" not in df.columns:
                seed_report['missing_fold_column_files'].append(filename)
                continue

            actual_folds = set(df["fold"].dropna().unique())
            missing = expected_folds - actual_folds
            if missing:
                seed_report['missing_folds_files'].append((filename, sorted(missing)))
            else:
                seed_report['files_all_folds_ok'].append(filename)
                print(f"✅ {seed}/{filename}: All fold values [1-5] present.")
        except Exception as e:
            seed_report['read_errors'].append((filename, str(e)))

    final_report[seed] = seed_report

# Print final summary
for seed, report in final_report.items():
    print(f"\n=== Seed {seed} Report ===")
    print(f"Status: {report['status']}")
    print(f"Number of CSV files: {report['num_csv_files']}")
    if report['missing_fold_column_files']:
        print(f"❌ Files missing 'fold' column: {report['missing_fold_column_files']}")
    if report['missing_folds_files']:
        print("❌ Files missing specific fold values:")
        for fname, missing in report['missing_folds_files']:
            print(f"   {fname}: Missing {missing}")
    if report['read_errors']:
        print("❌ Files with read errors:")
        for fname, err in report['read_errors']:
            print(f"   {fname}: {err}")
