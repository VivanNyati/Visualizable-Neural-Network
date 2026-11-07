# Verification

## Local verification on November 7, 2026

Windows CPU checks used Python 3.14.5, PyTorch 2.12.0+cpu, torchvision 0.27.0+cpu, Matplotlib 3.10.9, NumPy 2.4.6, and Pillow 12.2.0. The dependency ranges are in `requirements.txt`.
Python 3.11 and 3.12 are the versions targeted by the Windows/Linux CI matrix.
The local Python 3.14 run is an additional measured environment.

- All 18 unittest cases passed. They cover weight updates, checkpoint round trips,
  preprocessing, button and drawing events, run preservation and selection, and
  evaluation counts and exports. Tests use synthetic data and temporary models.
- `python train.py --no-ui --epochs 4 --run-dir runs/commit10-verification`
  completed 4 epochs and 3,440 optimizer steps on 55,000 training images,
  with 5,000 validation images, seed 42, and two CPU threads.
- Latest weights scored 98.39% on all 10,000 test images.
  Best validation accuracy was 98.48% at epoch 3.
- The included example was predicted as 7, with a 99.99%
  softmax score, using the latest weights.
- The saved-model evaluator reproduced the latest checkpoint's full-test result.
  Its PNG and JSON exports include confusion counts, per-digit metrics, and mistakes.

`training.png`, `prediction.png`, `evaluation.png`, and `example-metrics.json` in
this folder come from this measured run. The inspector screenshot includes the
Open image and Draw digit controls. Screenshots use Agg rendering. Desktop TkAgg
availability was checked by the doctor in the preceding diagnostics iteration.
The included digit is the first MNIST test image.

Generated datasets, run folders, and model weights are ignored by Git. After a
fresh clone, follow the README setup and training steps to create weights.

## Automated checks

`.github/workflows/tests.yml` installs CPU dependencies, runs headless diagnostics,
and executes `python -m unittest discover -v` on Windows and Linux with Python
3.11 and 3.12. It runs on pushes to main, pull requests, and manual dispatch.
MNIST downloads are unnecessary for these tests. Desktop GUI and GPU execution
remain outside the CI checks. The Actions page records each remote run's status.

The workflow follows the upstream [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) usage documentation.
