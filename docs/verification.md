# Verification

Verified on Windows with Python 3.12.14, PyTorch 2.15.0+cpu,
torchvision 0.29.2+cpu, and Matplotlib 3.11.2 (TkAgg desktop / Agg exports).

- `python -m unittest discover -v`: five tests passed, including checkpoint
  round-trip predictions, actual weight updates, logit reconstruction from
  contributions, input preprocessing, dashboard drawing, and filter switching.
- `python train.py --epochs 2 --no-ui`: 55,000 training examples, 5,000
  validation examples, 1,720 optimizer steps, seed 42, two CPU threads.
- Final validation accuracy: 98.06%; final test accuracy: 98.12% on all
  10,000 test examples. Training plus evaluation took about 81 seconds on the
  verification machine (excluding initial dataset download).
- `python predict.py examples/digit.png --no-ui --output runs/prediction.png`:
  predicted 7, matching the example label, with a 99.95% softmax score.
- Desktop TkAgg window drew successfully; Pause/Resume processed through
  its event loop and Stop set the expected state.
- Both exported dashboard layouts were visually inspected.

The sample image is the first MNIST test image. The screenshots and
`example-metrics.json` show the measured run, not a promised accuracy for other
training settings or custom images. GPU execution was not tested.

The local deliverable contains `checkpoints/mnist.pt` from this run; generated
checkpoints and downloaded datasets are ignored by Git. After a fresh clone,
run training before prediction.
