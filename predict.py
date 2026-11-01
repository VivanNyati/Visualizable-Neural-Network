"""Inspect one digit image: python predict.py path/to/digit.png."""
import argparse
from pathlib import Path
import torch
from model import DEFAULT_CHECKPOINT, get_device, load_checkpoint
from images import prepare_image, explain
from runs import RUNS_DIR, list_runs, select_checkpoint


def choose_image():
    try:
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
    except ImportError as error:
        raise ValueError('File picker unavailable; pass an image path on the command line.') from error
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            return askopenfilename(title='Choose an image containing one handwritten digit',
                filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff'),
                           ('All files', '*.*')])
        finally:
            root.destroy()
    except (tk.TclError, RuntimeError) as error:
        raise ValueError('File picker unavailable; pass an image path on the command line.') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, help='Omit to open a file picker')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint', type=Path, help='Explicit checkpoint, including older models')
    source.add_argument('--run', help='Saved run name, or latest (default)')
    parser.add_argument('--weights', choices=['latest', 'best'], default='latest')
    parser.add_argument('--runs-dir', type=Path, default=RUNS_DIR)
    parser.add_argument('--list-runs', action='store_true', help='List saved runs and exit')
    parser.add_argument('--polarity', choices=['auto', 'dark', 'light'], default='auto',
                        help='Ink color; auto detects background from image border')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--no-ui', action='store_true')
    parser.add_argument('--output', type=Path, help='Save the inspection dashboard as a PNG')
    args = parser.parse_args(argv)
    if args.list_runs:
        available = list_runs(args.runs_dir)
        for row in available:
            best = row.get('best')
            best_text = f'epoch {best["epoch"]}, {best["validation"]["accuracy"]:.2%}' if best else 'none'
            print(f'{row["name"]} | epochs: {row["epochs_completed"]} | '
                  f'status: {"stopped" if row["interrupted"] else "finished"} | best: {best_text}')
        if not available:
            print('No saved runs found. Run python train.py first.')
        return
    try:
        if args.checkpoint is not None:
            if args.weights != 'latest':
                parser.error('--weights selects weights within a run; use --run with --weights best.')
        elif args.run is None and not list_runs(args.runs_dir) and DEFAULT_CHECKPOINT.is_file() and args.weights == 'latest':
            args.checkpoint = DEFAULT_CHECKPOINT
        else:
            args.checkpoint = select_checkpoint(args.run or 'latest', args.weights, args.runs_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run python train.py first.')
    print(f'Using checkpoint: {args.checkpoint}')
    if args.no_ui and not args.image:
        parser.error('--no-ui requires an image path')
    if args.no_ui and not args.output:
        parser.error('--no-ui requires --output path/to/prediction.png')
    if args.no_ui:
        import matplotlib
        matplotlib.use('Agg')
    from dashboard import PredictionDashboard, export
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    device = get_device(args.device)
    try:
        path = args.image or choose_image()
        if not path:
            return
        model, _ = load_checkpoint(args.checkpoint, device)
        image = prepare_image(path, args.polarity).to(device)
        result = explain(model, image)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    view = PredictionDashboard(model, image, result, Path(path).name)
    digit = result['prediction']
    print(f'Prediction: {digit} | softmax score: {result["probabilities"][digit]:.2%}')
    if args.output:
        export(view.fig, args.output)
        print(f'Dashboard saved: {args.output}')
    if not args.no_ui:
        plt.show()
    plt.close(view.fig)


if __name__ == '__main__':
    main()
