"""Inspect one digit image: python predict.py path/to/digit.png."""
import argparse
from pathlib import Path
import torch
from model import DEFAULT_CHECKPOINT, get_device, load_checkpoint
from images import prepare_image, explain


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
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--polarity', choices=['auto', 'dark', 'light'], default='auto',
                        help='Ink color; auto detects background from image border')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--no-ui', action='store_true')
    parser.add_argument('--output', type=Path, help='Save the inspection dashboard as a PNG')
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run python train.py first.')
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
