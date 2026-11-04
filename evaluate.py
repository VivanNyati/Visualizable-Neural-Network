"""Evaluate saved MNIST weights: python evaluate.py --run latest --weights best."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import ROOT, DEFAULT_CHECKPOINT, get_device, load_checkpoint
from runs import RUNS_DIR, list_runs, select_checkpoint


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def assess(model, loader, device, mistake_limit=8):
    """Count every sample; retain only a bounded set of mistakes for display."""
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    mistakes, count, loss = [], 0, 0.
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(1)
            loss += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            pairs = (labels * 10 + predictions).cpu()
            confusion += torch.bincount(pairs, minlength=100).reshape(10, 10)
            for index in (predictions != labels).nonzero().flatten().tolist():
                if len(mistakes) >= mistake_limit:
                    break
                mistakes.append({'index': count + index, 'true': int(labels[index]),
                                 'predicted': int(predictions[index]),
                                 'image': images[index, 0].cpu().clone()})
            count += len(labels)
    if not count:
        raise ValueError('The evaluation dataset is empty.')
    per_digit = []
    for digit in range(10):
        samples = int(confusion[digit].sum())
        correct = int(confusion[digit, digit])
        per_digit.append({'digit': digit, 'samples': samples, 'correct': correct,
                          'accuracy': correct / samples if samples else None})
    return {'samples': count, 'accuracy': int(confusion.diag().sum()) / count,
            'loss': loss / count, 'confusion': confusion.tolist(), 'per_digit': per_digit}, mistakes


def render_report(report, mistakes):
    import numpy as np
    import matplotlib.pyplot as plt
    from dashboard import theme, BG, TEXT, MUTED, CYAN
    theme()
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.canvas.manager.set_window_title('Digit Lab | Model evaluation')
    fig.suptitle('DIGIT LAB  /  MODEL EVALUATION', x=.06, ha='left', fontsize=19, fontweight='bold')
    fig.text(.06, .925, f'{report["scope"]}  |  {report["samples"]:,} images  |  '
             f'Accuracy: {report["accuracy"]:.2%}  |  Loss: {report["loss"]:.4f}', color=TEXT)
    checkpoint = Path(report['checkpoint'])
    fig.text(.06, .895, f'Checkpoint: {checkpoint.parent.name} / {checkpoint.name}', color=MUTED, fontsize=9)
    grid = fig.add_gridspec(2, 2, left=.06, right=.95, top=.85, bottom=.07,
                           hspace=.42, wspace=.35, height_ratios=[1.2, 1])
    ax = fig.add_subplot(grid[0, 0])
    matrix = np.array(report['confusion'])
    artist = ax.imshow(matrix, cmap='Blues', interpolation='nearest')
    fig.colorbar(artist, ax=ax, label='Images', fraction=.046, pad=.04)
    ax.set(title='Confusion matrix (counts)', xlabel='Predicted digit', ylabel='True digit',
           xticks=range(10), yticks=range(10))
    for row in range(10):
        for col in range(10):
            if matrix[row, col]:
                ax.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=7,
                        color='white' if matrix[row, col] > matrix.max() / 2 else 'black')
    ax = fig.add_subplot(grid[0, 1])
    accuracies = [100 * row['accuracy'] if row['accuracy'] is not None else 0
                  for row in report['per_digit']]
    ax.bar(range(10), accuracies, color=CYAN)
    ax.set(title='Accuracy per true digit', xlabel='Digit (sample count below)',
           ylabel='Correct (%)', ylim=(0, 112), xticks=range(10),
           xticklabels=[f'{row["digit"]}\n{row["samples"]}' for row in report['per_digit']])
    ax.tick_params(axis='x', labelsize=8)
    for index, row in enumerate(report['per_digit']):
        ax.text(index, accuracies[index] + 2, f'{accuracies[index]:.1f}%' if row['samples'] else 'N/A',
                ha='center', fontsize=8)
    examples = grid[1, :].subgridspec(2, 4, wspace=.25, hspace=.55)
    for index in range(8):
        ax = fig.add_subplot(examples[index // 4, index % 4])
        ax.set(xticks=[], yticks=[])
        if index < len(mistakes):
            item = mistakes[index]
            ax.imshow(item['image'].numpy() * .5 + .5, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'#{item["index"]}: true {item["true"]} / predicted {item["predicted"]}', fontsize=9)
        else:
            ax.set_axis_off()
    fig.text(.06, .025, 'First eight mistakes in dataset order.' if mistakes else
             'No mistakes in the evaluated images.', color=MUTED, fontsize=9)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--run', help='Saved run name or latest (default)')
    parser.add_argument('--weights', choices=['latest', 'best'], default='latest')
    parser.add_argument('--runs-dir', type=Path, default=RUNS_DIR)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--max-eval-samples', type=positive)
    parser.add_argument('--batch-size', type=positive, default=256)
    parser.add_argument('--threads', type=positive, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--output-dir', type=Path, help='Folder for evaluation.json and evaluation.png')
    parser.add_argument('--no-ui', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.checkpoint is not None:
            if args.weights != 'latest':
                raise ValueError('Use --run with --weights best, or pass the best checkpoint directly.')
            checkpoint = args.checkpoint
        elif args.run is None and not list_runs(args.runs_dir) and DEFAULT_CHECKPOINT.is_file() and args.weights == 'latest':
            checkpoint = DEFAULT_CHECKPOINT
        else:
            checkpoint = select_checkpoint(args.run or 'latest', args.weights, args.runs_dir)
        device = get_device(args.device)
        torch.set_num_threads(args.threads)
        model, _ = load_checkpoint(checkpoint, device)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
        dataset = datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
        available = len(dataset)
        if args.max_eval_samples:
            dataset = Subset(dataset, range(min(args.max_eval_samples, available)))
        report, mistakes = assess(model, DataLoader(dataset, batch_size=args.batch_size), device)
        report.update({'checkpoint': str(checkpoint.resolve()), 'available_samples': available,
                       'scope': 'Full test set' if len(dataset) == available else 'Test subset',
                       'mistakes': [{key: value for key, value in item.items() if key != 'image'}
                                    for item in mistakes]})
        output = args.output_dir or RUNS_DIR / 'evaluations' / (
            datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
        output.mkdir(parents=True, exist_ok=True)
        if any((output / name).exists() for name in ('evaluation.json', 'evaluation.png')):
            raise ValueError('Evaluation output already exists. Choose a new --output-dir.')
        if args.no_ui:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from dashboard import export
        fig = render_report(report, mistakes)
        try:
            export(fig, output / 'evaluation.png')
            (output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'{report["scope"]}: {report["samples"]:,} images | Accuracy: {report["accuracy"]:.2%}')
            print(f'Evaluation saved: {output.resolve()}')
            if not args.no_ui:
                plt.show()
        finally:
            plt.close(fig)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
