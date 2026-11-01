"""Run folders and lightweight summaries; listing never loads model weights."""
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

from model import ROOT

RUNS_DIR = ROOT / 'runs'


def create_run(directory=None, root=RUNS_DIR):
    explicit = directory is not None
    directory = Path(directory) if directory else Path(root) / (
        datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=explicit)
    for name in ('run.json', 'metrics.json', 'training.png', 'mnist.pt', 'mnist.best.pt'):
        if (directory / name).exists():
            raise ValueError(f'Run folder already contains results: {directory}. Choose a new --run-dir.')
    return directory.resolve()


def write_summary(directory, created, checkpoint, best, metadata):
    summary = {'name': directory.name, 'created': created,
               'checkpoint': str(checkpoint.resolve()) if metadata['steps'] else None,
               'best_checkpoint': best['checkpoint'] if best else None,
               'epochs_completed': metadata['epochs_completed'],
               'steps': metadata['steps'], 'interrupted': metadata['interrupted'],
               'test': metadata['test'], 'best': best}
    temporary = directory / 'run.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temporary.replace(directory / 'run.json')


def list_runs(root=RUNS_DIR):
    results = []
    for path in Path(root).glob('*/run.json'):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(row, dict) or not isinstance(row.get('created'), str):
                continue
            if not isinstance(row.get('epochs_completed'), int) or not isinstance(row.get('interrupted'), bool):
                continue
            best = row.get('best')
            if best is not None and (not isinstance(best, dict) or
                    not isinstance(best.get('epoch'), int) or
                    not isinstance(best.get('validation'), dict) or
                    not isinstance(best['validation'].get('accuracy'), (int, float))):
                continue
            checkpoint = row.get('checkpoint')
            if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
                continue
            row['name'] = path.parent.name
            results.append(row)
        except (OSError, ValueError):
            continue
    return sorted(results, key=lambda row: row['created'], reverse=True)


def select_checkpoint(name='latest', weights='latest', root=RUNS_DIR):
    available = list_runs(root)
    selected = next((row for row in available if name == 'latest' or row['name'] == name), None)
    if selected is None:
        raise ValueError(f'No saved run found for {name!r}. Use --list-runs or train a model first.')
    checkpoint = selected.get('best_checkpoint' if weights == 'best' else 'checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError(f'Run {selected["name"]} has no {weights} checkpoint. Try --weights latest.')
    return Path(checkpoint)
