"""Exercise saving on early stop and normal completion without downloading data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import Dataset

from dashboard import TrainingDashboard
from model import load_checkpoint
from train import evaluate, main


class TinyDigits(Dataset):
    def __len__(self):
        return 60000

    def __getitem__(self, index):
        return torch.ones(1, 28, 28), index % 10


class TrainingSaveTests(unittest.TestCase):
    def test_completion_and_stop_save_the_expected_checkpoint(self):
        for stop_at in [None, 'before_first_batch', 'last_batch', 'validation']:
            with self.subTest(stop_at=stop_at), tempfile.TemporaryDirectory() as directory:
                views = []

                class View(TrainingDashboard):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        views.append(self)

                    def update(self, *args, **kwargs):
                        super().update(*args, **kwargs)
                        if stop_at == 'last_batch':
                            self.stop(None)

                    def pump(self):
                        if stop_at == 'before_first_batch':
                            self.stop(None)

                def evaluate_and_stop(*args, **kwargs):
                    result = evaluate(*args, **kwargs)
                    if stop_at == 'validation':
                        views[0].stop(None)
                    return result

                checkpoint = Path(directory) / 'model.pt'
                with patch('torchvision.datasets.MNIST', return_value=TinyDigits()), \
                        patch('dashboard.TrainingDashboard', View), \
                        patch('train.evaluate', side_effect=evaluate_and_stop):
                    main(['--no-ui', '--epochs', '1', '--max-train-samples', '64',
                          '--max-eval-samples', '16', '--checkpoint', str(checkpoint),
                          '--run-dir', directory])
                metadata = json.loads((Path(directory) / 'metrics.json').read_text())
                trained = stop_at != 'before_first_batch'
                self.assertEqual(metadata['interrupted'], stop_at is not None)
                self.assertEqual(metadata['steps'], int(trained))
                self.assertEqual(metadata['epochs_completed'],
                                 int(stop_at in [None, 'validation']))
                self.assertTrue(views[0].finished)
                self.assertEqual(views[0].validation_losses,
                                 [row['validation']['loss'] for row in metadata['history']])
                self.assertEqual(views[0].validation_accuracies,
                                 [100 * row['validation']['accuracy'] for row in metadata['history']])
                if trained:
                    _, saved = load_checkpoint(checkpoint, torch.device('cpu'))
                    self.assertEqual(saved, metadata)
                    self.assertIn(str(checkpoint), views[0].save_status.get_text())
                else:
                    self.assertFalse(checkpoint.exists())
                    self.assertIn('No model saved', views[0].save_status.get_text())


if __name__ == '__main__':
    unittest.main()
