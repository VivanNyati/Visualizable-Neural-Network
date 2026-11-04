# Improvement roadmap

Ten focused commits, in order. Each should deliver working behavior, relevant
checks, and updated usage instructions. Complete one per requested iteration;
later items may be adjusted as user feedback reveals better priorities.

| # | Planned commit | Improvement and acceptance checks | Status |
| --- | --- | --- | --- |
| 1 | Fix training stop, save, and completed controls | Show saving feedback and the checkpoint location; keep saved results open; turn the action into Close; disable Pause after completion. Check normal completion, stopping while paused, stopping on the last batch or during validation, and stopping before any weight update. | Implemented |
| 2 | Show validation progress and epoch boundaries | Add validation metrics and visible epoch boundaries so the resetting training averages do not look like unexplained jumps. Label training and validation separately. | Implemented |
| 3 | Preserve the best validation model | Save the best validation checkpoint alongside the latest weights and report its epoch. Select models using validation only; keep the test set for final evaluation. | Implemented |
| 4 | Organize training runs and model selection | Give runs distinct folders and summaries; let prediction select a saved run without manually typing a checkpoint path. Check that a new run preserves earlier models. | Implemented |
| 5 | Load another image from the inspector | Add an Open image control that refreshes the existing inspector. Handle cancellation and invalid images without losing the loaded model or closing the window. | Implemented |
| 6 | Draw a digit and test it | Add a drawing canvas with Clear and Predict, using the same preprocessing as image files. Show the normalized input so users can understand unexpected results. | Implemented |
| 7 | Add a saved-model evaluation view | Evaluate an existing checkpoint with a confusion matrix, per-digit accuracy, and example mistakes. Separate full-test and subset results and avoid retraining. | Implemented |
| 8 | Improve dashboard layout and readability | Make spacing, labels, color legends, and sample captions readable at common window sizes. Inspect both training and prediction views at multiple sizes. | Implemented |
| 9 | Add setup diagnostics and actionable errors | Add a diagnostic command for dependencies, GUI support, device availability, data, and checkpoints. Explain how to fix failures and document a fresh-clone walkthrough. | Planned |
| 10 | Automate checks and refresh project documentation | Add GitHub Actions for CPU tests on Windows and Linux, document supported versions, and update screenshots and the quick start to match the shipped workflow. | Planned |

## Commit 1 validation

- Exercise real Matplotlib button events for Pause, Stop & save, and Close.
- Train on a tiny synthetic dataset to verify saved weights and metadata for
  completed and interrupted runs, without downloading MNIST in the test suite.
- Check that stopping before the first optimizer step creates no checkpoint.
- Run the full unittest suite and inspect the desktop preview.
