import os
import json
import argparse
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set

def find_completed_experiments(input_dir: str) -> List[str]:
    """Find directories containing completed experiments"""
    completed_dirs = []
    for root, _, files in os.walk(input_dir):
        if 'done' in files and 'args.json' in files and 'result.json' in files:
            completed_dirs.append(root)
    return completed_dirs

def read_experiment_data(exp_dir: str) -> Tuple[Dict, float]:
    """Read experiment parameters and results from a directory"""
    # Read args.json
    with open(os.path.join(exp_dir, 'args.json'), 'r') as f:
        args = json.load(f)
    
    # Read result.json
    with open(os.path.join(exp_dir, 'result.json'), 'r') as f:
        test_acc = json.load(f)['test_acc']
    
    return args, test_acc

def get_unique_values(all_args: List[Dict], key: str) -> Set[str]:
    """Extract unique values for a given key from all experiments"""
    return set(args[key] for args in all_args)

class MethodKey:
    """Class to handle method key creation and formatting"""
    def __init__(self, args: Dict):
        self.method_name = args.get('method', None)
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.optim = args['optim']
        self.lr = args['learning_rate']

    def get_main_config(self) -> str:
        """Get main method configuration (method and prompt size)"""
        return f"{self.method_name}"

    def get_hyper_params(self) -> str:
        """Get hyperparameter configuration"""
        hyper_params = f'e{self.epochs}_bs{self.batch_size}_{self.optim}{self.lr}'
        
        return f"{hyper_params}"

    def get_sorting_key(self) -> Tuple:
        """Create a sorting key for consistent ordering"""
        return (
            # Primary sort by method name
            self.method_name, 
            # Secondary sort by epochs
            self.epochs, 
            # Tertiary sort by batch size
            self.batch_size, 
            # Quaternary sort by optimizer
            self.optim, 
        )

    def __hash__(self):
        return hash((self.method_name, self.epochs, 
                    self.batch_size, self.optim))

    def __eq__(self, other):
        if not isinstance(other, MethodKey):
            return False
        return (self.method_name == other.method_name and 
                self.epochs == other.epochs and
                self.batch_size == other.batch_size and
                self.optim == other.optim)

def collect_results(args) -> Tuple[List[str], List[str], Dict]:
    """Collect and organize experiment results"""
    completed_dirs = find_completed_experiments(args.dir)
    print(f"Found {len(completed_dirs)} completed experiments")
    
    # Read all experiment data
    all_data = []
    for exp_dir in completed_dirs:
        args, test_acc = read_experiment_data(exp_dir)
        all_data.append((args, test_acc))
    
    # Get unique datasets and models
    all_args = [args for args, _ in all_data]
    datasets = sorted(get_unique_values(all_args, 'dataset'))
    models = sorted(get_unique_values(all_args, 'model'))
    
    # Organize results by model, method, and dataset
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for args, test_acc in all_data:
        model = args['model']
        dataset = args['dataset']
        method_key = MethodKey(args)
        results[model][method_key][dataset].append(test_acc)
    
    return datasets, models, results, len(completed_dirs)

def format_mean_std_error(values: List[float]) -> str:
    """Format mean and standard error in LaTeX format"""
    if not values:
        return '-'
    mean = np.mean(values)
    std_error = np.std(values) / np.sqrt(len(values))
    return f"{mean:.2f} $\\pm$ {std_error:.2f}"

def generate_latex_table(model: str, datasets: List[str], 
                        model_results: Dict) -> str:
    """Generate LaTeX table for a specific model"""
    # Calculate table width based on number of datasets
    table_width = len(datasets) + 3  # method + hyperparams + datasets + average columns

    # Start table
    latex = f"\\begin{{table}}[h]\n"
    latex += "\\centering\n\\small\n"
    latex += "\\begin{adjustbox}{max width=\\textwidth}\n"
    latex += f"\\begin{{tabular}}{{ll{'c' * (table_width-2)}}}\n"
    latex += "\\toprule\n"
    
    # Header row 
    latex += "Method & Hyperparameters & " + \
             " & ".join(datasets) + " & Average \\\\\n"
    latex += "\\midrule\n"
    
    # Group results by main configuration
    grouped_results = defaultdict(list)
    for method_key in model_results.keys():
        main_config = method_key.get_main_config()
        grouped_results[main_config].append(method_key)
    
    # Results for each method group
    first_in_group = True
    for main_config, method_keys in sorted(grouped_results.items()):
        # Sort method keys using the new sorting key method
        sorted_method_keys = sorted(method_keys, key=lambda x: str(x.get_sorting_key()))
        
        for method_key in sorted_method_keys:
            dataset_results = model_results[method_key]
            row_values = []
            dataset_means = []
            
            # Collect results for each dataset
            for dataset in datasets:
                values = dataset_results.get(dataset, [])
                if values:
                    dataset_means.append(np.mean(values))
                formatted = format_mean_std_error(values)
                row_values.append(formatted)
            
            # Calculate average across datasets
            if dataset_means:
                overall_mean = np.mean(dataset_means)
                overall_std = np.std(dataset_means) / np.sqrt(len(dataset_means))
                overall_result = f"{overall_mean:.2f} $\\pm$ {overall_std:.2f}"
            else:
                overall_result = "-"
            
            # Add row to table
            if first_in_group:
                main_method = main_config
                first_in_group = False
            else:
                main_method = "\\quad"  # Indent for subsequent rows
            
            row = f"{main_method} & {method_key.get_hyper_params()} & " + \
                    f"{' & '.join(row_values)} & {overall_result} \\\\\n"
            row = row.replace("_", "\\_")
            latex += row
        
        # Add small space between groups
        if not first_in_group:
            latex += "\\addlinespace[2pt]\n"
            first_in_group = True
    
    # Table footer
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{adjustbox}\n"
    latex += f"\\caption{{Results for {model}}}\n".replace("_", "\\_")
    latex += f"\\label{{tab:results_{model}}}\n"
    latex += "\\end{table}\n\n"
    
    return latex

def main():
    parser = argparse.ArgumentParser(description='Collect and summarize experiment results')
    parser.add_argument('--dir', required=True, 
                       help='Directory containing experiment results')
    args = parser.parse_args()
    
    # Collect results
    datasets, models, results, num_results = collect_results(args)   
    output_path = os.path.join(args.dir, 'results.tex')
    
    # Generate LaTeX document and save to file
    with open(output_path, 'w') as f:
        # Document header
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{adjustbox}\n")
        f.write("\\usepackage[margin=20truemm]{geometry}\n")
        f.write("\\begin{document}\n\n")
        f.write(f"Found {num_results} completed experiments\n")
        
        # Generate tables for each model
        for model in models:
            f.write(f"\\section{{{model}}}\n\n".replace("_", "\\_"))
            latex_table = generate_latex_table(model, datasets, results[model])
            f.write(latex_table)
        
        # Document footer
        f.write("\\end{document}\n")
    
    print(f"Results have been saved to: {output_path}")

if __name__ == "__main__":
    main()
