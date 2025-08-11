import os
import re
import glob
import statistics
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Regex to extract metrics from each file
METRIC_REGEX = re.compile(r"Precision: ([\d.]+)\s+Recall: ([\d.]+)\s+F1: ([\d.]+).*?Loss: ([\d.]+)")

# Directory containing seed_* folders
BASE_DIR = "baselines/results/cd/"  # <-- Change this

# Stores data like:
# data[metric][config][ratio] = average value across seeds
aggregated_data = defaultdict(lambda: defaultdict(dict))

# Temporary storage before averaging across seeds
temp_data_by_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

logging.info("Starting to process seed directories...")

# Traverse each seed directory
for seed_dir in sorted(glob.glob(os.path.join(BASE_DIR, "seed_*"))):
    logging.info(f"Processing directory: {seed_dir}")
    for file_path in glob.glob(os.path.join(seed_dir, "*.txt")):
        with open(file_path, 'r') as f:
            content = f.read()

        match = METRIC_REGEX.search(content)
        if not match:
            logging.warning(f"Could not extract metrics from: {file_path}")
            continue

        precision, recall, f1, loss = map(float, match.groups())
        logging.debug(f"Parsed metrics from {file_path} — P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}")

        # Parse config from filename
        filename = os.path.basename(file_path)
        parts = filename.split("_classification_ratio_")[1].split("_fold_")[0].split("_")
        frac = parts[0]
        local_info = parts[1]
        global_info = parts[2]
        model_name = "_".join(parts[3:])

        config_str = f"({local_info},{global_info},{model_name})"
        temp_data_by_seed[config_str][frac]['precision'].append(precision)
        temp_data_by_seed[config_str][frac]['recall'].append(recall)
        temp_data_by_seed[config_str][frac]['f1'].append(f1)
        temp_data_by_seed[config_str][frac]['loss'].append(loss)

# Average across seeds (each list should have 15 values = 3 seeds * 5 folds)
logging.info("Averaging metrics across folds and seeds...")
for config in temp_data_by_seed:
    for frac in temp_data_by_seed[config]:
        for metric in ['precision', 'recall', 'f1', 'loss']:
            values = temp_data_by_seed[config][frac][metric]
            if values:
                avg = sum(values) / len(values)
                aggregated_data[metric][config][frac] = avg
                logging.debug(f"Averaged {metric} for {config} ratio {frac}: {avg:.4f}")

# LaTeX table generation
def generate_combined_latex_table(data, ratios):
    logging.info("Generating LaTeX table...")
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Loss']
    metric_keys = ['precision', 'recall', 'f1', 'loss']

    header = r"\begin{table*}[ht]\n" + \
             r"\centering\n" + \
             r"\begin{tabular}{ll" + "c" * len(ratios) + r"}\n" + \
             r"\hline\n"

    latex_table = header
    latex_table += r"\textbf{Metric} & \textbf{Config} & " + " & ".join([
        rf"\textbf{{{float(r) * 100:.1f}\%}}" for r in ratios]) + r" \\\n\hline\n"

    config_labels = {
        '(None,None,bert-base-uncased)': '(i) Utterance',
        '(intentions,None,bert-base-uncased)': '(ii) Intentions',
        '(None,scd,summary_bert-base-uncased)': '(iii) SCD Summary',
        '(None,scm,summary_bert-base-uncased)': '(iv) SCM Summary',
        '(None,traditional,summary_bert-base-uncased)': '(v) Traditional Summary',
        '(intentions,scd,summary_bert-base-uncased)': '(vi) Intentions + SCD',
        '(intentions,scm,summary_bert-base-uncased)': '(vii) Intentions + SCM',
        '(intentions,traditional,summary_bert-base-uncased)': '(viii) Intentions + Traditional'
    }

    baseline_type = '(None,None,bert-base-uncased)'

    for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        metric_data = data[metric_key]
        config_types = sorted(metric_data.keys(), key=lambda x: config_labels.get(x, x))

        baseline_values = {
            ratio: metric_data.get(baseline_type, {}).get(ratio)
            for ratio in ratios
        }

        first_config = True
        for config_type in config_types:
            label = config_labels.get(config_type, config_type)
            row = ""
            if first_config:
                row += rf"\multirow{{{len(config_types)}}}{{*}}{{\textbf{{{metric_name}}}}} & {label} & "
                first_config = False
            else:
                row += f" & {label} & "

            cells = []
            for ratio in ratios:
                value = metric_data[config_type].get(ratio)
                if value is None:
                    cells.append('-')
                else:
                    formatted = f"{value:.4f}"
                    base = baseline_values.get(ratio)
                    if base is None or config_type == baseline_type:
                        cells.append(formatted)
                    else:
                        if value == base:
                            color = r'\cellcolor{yellow!25}'
                        elif value > base:
                            color = r'\cellcolor{green!25}'
                        else:
                            color = r'\cellcolor{red!25}'
                        cells.append(rf"{color}{formatted}")
            row += " & ".join(cells) + r" \\\n"
            latex_table += row

        if i < len(metric_names) - 1:
            latex_table += r"\midrule\n"

    latex_table += r"""\hline
\end{tabular}
\caption{\textbf{SFT: }Performance metrics across different configurations on dataset}
\label{tab:combined_metrics}
\end{table*}"""

    logging.info("LaTeX table generation complete.")
    print(latex_table)
    return latex_table

# Call the table generator
ratios_present = sorted({frac for metric_data in aggregated_data['precision'].values() for frac in metric_data.keys()}, key=float)
logging.info(f"Ratios detected: {ratios_present}")
generate_combined_latex_table(aggregated_data, ratios_present)
