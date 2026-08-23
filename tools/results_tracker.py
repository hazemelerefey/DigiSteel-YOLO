"""
Master Results Tracker for DigiSteel-YOLO
Automatically collects and tracks all experiment results.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ResultsTracker:
    """Track all experiment results in one place."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.results_file = self.project_root / 'evals' / 'master_results.json'
        self.results = self._load_results()
    
    def _load_results(self) -> dict:
        """Load existing results."""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {
            'metadata': {
                'project': 'DigiSteel-YOLO',
                'dataset': 'NEU-DET',
                'target_metric': 'mAP50-95',
                'target_value': 0.83,
                'last_updated': None,
            },
            'experiments': {},
            'best_results': {
                'mAP50': {'value': 0, 'experiment': None},
                'mAP50_95': {'value': 0, 'experiment': None},
            }
        }
    
    def _save_results(self):
        """Save results to file."""
        self.results['metadata']['last_updated'] = datetime.now().isoformat()
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def add_experiment(self, exp_id: str, name: str, description: str = ""):
        """Add a new experiment."""
        if exp_id not in self.results['experiments']:
            self.results['experiments'][exp_id] = {
                'name': name,
                'description': description,
                'created': datetime.now().isoformat(),
                'runs': [],
                'best_map50': 0,
                'best_map50_95': 0,
            }
            self._save_results()
    
    def add_run(self, exp_id: str, results: dict, config: dict = None):
        """
        Add a training run result.
        
        Args:
            exp_id: Experiment ID
            results: Dict with mAP50, mAP50_95, precision, recall, per_class_ap50
            config: Training configuration used
        """
        if exp_id not in self.results['experiments']:
            raise ValueError(f"Experiment {exp_id} not found. Call add_experiment first.")
        
        run = {
            'timestamp': datetime.now().isoformat(),
            'mAP50': results.get('map50', results.get('mAP50', 0)),
            'mAP50_95': results.get('map50_95', results.get('mAP50_95', 0)),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'per_class_ap50': results.get('per_class_ap50', {}),
            'config': config or {},
        }
        
        exp = self.results['experiments'][exp_id]
        exp['runs'].append(run)
        
        # Update best results
        if run['mAP50'] > exp['best_map50']:
            exp['best_map50'] = run['mAP50']
        if run['mAP50_95'] > exp['best_map50_95']:
            exp['best_map50_95'] = run['mAP50_95']
        
        # Update global best
        if run['mAP50'] > self.results['best_results']['mAP50']['value']:
            self.results['best_results']['mAP50'] = {
                'value': run['mAP50'],
                'experiment': exp_id,
                'timestamp': run['timestamp'],
            }
        if run['mAP50_95'] > self.results['best_results']['mAP50_95']['value']:
            self.results['best_results']['mAP50_95'] = {
                'value': run['mAP50_95'],
                'experiment': exp_id,
                'timestamp': run['timestamp'],
            }
        
        self._save_results()
        return run
    
    def load_from_evals(self):
        """Load all existing results from evals/ directory."""
        evals_dir = self.project_root / 'evals'
        if not evals_dir.exists():
            print("No evals directory found")
            return
        
        loaded = 0
        for json_file in evals_dir.glob('*.json'):
            if json_file.name == 'master_results.json':
                continue
            
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Extract experiment ID from filename
                exp_id = json_file.stem.replace('_results', '').replace('_summary', '')
                
                # Check if it has results
                map50 = data.get('map50', data.get('mAP50', data.get('baseline_map50', None)))
                if map50 is None:
                    continue
                
                # Add experiment if not exists
                if exp_id not in self.results['experiments']:
                    self.add_experiment(exp_id, exp_id.replace('_', ' ').title())
                
                # Add run
                self.add_run(exp_id, data)
                loaded += 1
                
            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")
        
        print(f"Loaded {loaded} results from evals/")
    
    def load_from_experiments(self):
        """Load results from experiments/ directory."""
        exp_dir = self.project_root / 'experiments'
        if not exp_dir.exists():
            print("No experiments directory found")
            return
        
        loaded = 0
        for exp_folder in sorted(exp_dir.iterdir()):
            if not exp_folder.is_dir():
                continue
            
            results_dir = exp_folder / 'results'
            if not results_dir.exists():
                continue
            
            for json_file in results_dir.glob('*.json'):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    map50 = data.get('map50', data.get('mAP50', data.get('baseline_map50', None)))
                    if map50 is None:
                        continue
                    
                    exp_id = exp_folder.name
                    if exp_id not in self.results['experiments']:
                        self.add_experiment(exp_id, exp_id.replace('_', ' ').title())
                    
                    self.add_run(exp_id, data)
                    loaded += 1
                    
                except Exception as e:
                    print(f"  Error loading {json_file.name}: {e}")
        
        print(f"Loaded {loaded} results from experiments/")
    
    def print_summary(self):
        """Print summary of all experiments."""
        print("\n" + "=" * 80)
        print("📊 MASTER RESULTS TRACKER — DigiSteel-YOLO")
        print("=" * 80)
        
        # Best results
        best = self.results['best_results']
        print(f"\n🏆 BEST RESULTS:")
        print(f"  mAP50:    {best['mAP50']['value']:.3f} ({best['mAP50']['experiment']})")
        print(f"  mAP50-95: {best['mAP50_95']['value']:.3f} ({best['mAP50_95']['experiment']})")
        
        # All experiments
        print(f"\n📋 ALL EXPERIMENTS:")
        print("-" * 80)
        print(f"  {'ID':<35s} {'Runs':>5s} {'Best mAP50':>10s} {'Best mAP50-95':>13s}")
        print("-" * 80)
        
        for exp_id, exp in sorted(self.results['experiments'].items()):
            n_runs = len(exp['runs'])
            map50 = exp['best_map50']
            map5095 = exp['best_map50_95']
            
            # Highlight best
            marker = ""
            if map50 == best['mAP50']['value']:
                marker = " 🏆"
            elif map5095 == best['mAP50_95']['value']:
                marker = " ⭐"
            
            print(f"  {exp_id:<35s} {n_runs:>5d} {map50:>10.3f} {map5095:>13.3f}{marker}")
        
        print("-" * 80)
        print(f"  Total: {len(self.results['experiments'])} experiments")
        
        # Target progress
        target = self.results['metadata']['target_value']
        current = best['mAP50_95']['value']
        progress = current / target * 100
        print(f"\n🎯 TARGET: {target:.1%} mAP50-95")
        print(f"  Current: {current:.3f} ({progress:.1f}% of target)")
        print(f"  Gap: {target - current:.3f}")
    
    def get_per_class_summary(self) -> pd.DataFrame:
        """Get per-class AP summary across all experiments."""
        import pandas as pd
        
        rows = []
        for exp_id, exp in self.results['experiments'].items():
            if not exp['runs']:
                continue
            best_run = max(exp['runs'], key=lambda r: r['mAP50'])
            per_class = best_run.get('per_class_ap50', {})
            per_class['experiment'] = exp_id
            per_class['mAP50'] = best_run['mAP50']
            per_class['mAP50_95'] = best_run['mAP50_95']
            rows.append(per_class)
        
        return pd.DataFrame(rows)


def main():
    """Main entry point."""
    tracker = ResultsTracker()
    tracker.load_from_evals()
    tracker.load_from_experiments()
    tracker.print_summary()


if __name__ == "__main__":
    main()
