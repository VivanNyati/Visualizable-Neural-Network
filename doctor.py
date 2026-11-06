"""Check Digit Lab's local setup before training or opening a dashboard."""
import argparse
from importlib import import_module
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def check_display(headless):
    import matplotlib
    if headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.ioff():
        figure = plt.figure()
        try:
            backend = matplotlib.get_backend()
            if not headless and not figure.canvas.required_interactive_framework:
                raise RuntimeError(f'{backend} does not open desktop windows')
            figure.canvas.draw()
            return backend
        finally:
            plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-ui', action='store_true', help='Check PNG rendering with Agg')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--checkpoint', type=Path, help='Check this file instead of the latest saved model')
    args = parser.parse_args(argv)

    failures = 0

    def fail(label, error, fix):
        nonlocal failures
        failures += 1
        print(f'FAIL {label}: {error}\n     {fix}')

    print(f'Python {sys.version.split()[0]} ({sys.executable})')
    modules = {}
    for package, name in [('torch', 'torch'), ('torchvision', 'torchvision'),
                          ('matplotlib', 'matplotlib'), ('numpy', 'numpy'), ('Pillow', 'PIL')]:
        try:
            module = import_module(name)
            modules[name] = module
            print(f'OK   {package}: {module.__version__}')
        except Exception as error:
            fix = 'Activate the project environment and run python -m pip install -r requirements.txt.'
            if name in ('torch', 'torchvision'):
                fix += ' Install matching torch and torchvision builds if both packages are present.'
            fail(package, error, fix)

    if 'matplotlib' in modules:
        try:
            print(f'OK   Display: {check_display(args.no_ui)}')
        except Exception as error:
            fail('Display', error, 'Use a desktop Python with Tk support, or use --no-ui for PNG exports.')
    else:
        print('SKIP Display: Matplotlib is unavailable.')

    if 'torch' in modules:
        from model import DEFAULT_CHECKPOINT, get_device, load_checkpoint
        from runs import list_runs

        try:
            device = get_device(args.device)
            modules['torch'].ones(1, device=device).cpu()
            print(f'OK   Device: {device}')
        except Exception as error:
            fail('Device', error, 'Try --device cpu or install a PyTorch build for the requested device.')

        try:
            checkpoint = args.checkpoint
            if checkpoint is None:
                available = list_runs(args.runs_dir)
                if available:
                    checkpoint = Path(available[0]['checkpoint'])
                elif DEFAULT_CHECKPOINT.is_file():
                    checkpoint = DEFAULT_CHECKPOINT
            if checkpoint is None:
                print('INFO Checkpoint: no saved model. Run python train.py to create one.')
            else:
                load_checkpoint(checkpoint, 'cpu')
                print(f'OK   Checkpoint: {checkpoint.resolve()}')
        except Exception as error:
            fail('Checkpoint', error, 'Pass --checkpoint with a compatible saved model, or train a new one.')
    else:
        print('SKIP Device and checkpoint: PyTorch is unavailable.')

    if 'torchvision' in modules:
        from torchvision.datasets import MNIST

        for label, train in [('Training data', True), ('Test data', False)]:
            try:
                dataset = MNIST(args.data_dir, train=train, download=False)
                print(f'OK   {label}: {len(dataset):,} images in {args.data_dir}')
            except RuntimeError as error:
                if 'Dataset not found' in str(error):
                    print(f'INFO {label}: MNIST is missing in {args.data_dir}. Training downloads it on first run.')
                else:
                    fail(label, error, 'Use --data-dir with a fresh folder to download a clean copy.')
            except Exception as error:
                fail(label, error, 'Use --data-dir with a fresh folder to download a clean copy.')
    else:
        print('SKIP Data: torchvision is unavailable.')
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
