"""Known prediction counts and saved-checkpoint evaluation without training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate import assess, main, render_report
from model import DigitNet, save_checkpoint


class KnownPredictions(torch.nn.Module):
    def forward(self, images):
        return torch.nn.functional.one_hot(images[:, 0, 0, 0].long(), 10).float() * 5


class EvaluationTests(unittest.TestCase):
    def dataset(self):
        images = torch.zeros(5, 1, 28, 28)
        images[:, 0, 0, 0] = torch.tensor([0, 1, 1, 0, 1])
        return TensorDataset(images, torch.tensor([0, 0, 1, 2, 2]))

    def test_known_confusion_counts_and_bounded_mistakes(self):
        report, mistakes = assess(KnownPredictions(), DataLoader(self.dataset(), batch_size=2),
                                  'cpu', mistake_limit=2)
        self.assertEqual(report['samples'], 5)
        self.assertEqual(report['accuracy'], .4)
        self.assertEqual([row[:3] for row in report['confusion'][:3]], [[1, 1, 0], [0, 1, 0], [1, 1, 0]])
        self.assertEqual([row['accuracy'] for row in report['per_digit'][:4]], [.5, 1., 0., None])
        self.assertEqual([item['index'] for item in mistakes], [1, 3])
        self.assertGreater(report['loss'], 0)
        with self.assertRaisesRegex(ValueError, 'empty'):
            assess(KnownPredictions(), [], 'cpu')

    def test_cli_full_and_subset_exports_preserve_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            model = DigitNet()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            save_checkpoint(model, checkpoint, {})
            original = checkpoint.read_bytes()
            for limit, scope, count in [(None, 'Full test set', 5), (3, 'Test subset', 3)]:
                output = root / str(count)
                arguments = ['--no-ui', '--checkpoint', str(checkpoint), '--output-dir', str(output)]
                if limit:
                    arguments += ['--max-eval-samples', str(limit)]
                with patch('evaluate.datasets.MNIST', return_value=self.dataset()) as dataset:
                    main(arguments)
                    self.assertFalse(dataset.call_args.kwargs['train'])
                report = json.loads((output / 'evaluation.json').read_text())
                self.assertEqual(report['scope'], scope)
                self.assertEqual(report['samples'], count)
                self.assertEqual(sum(map(sum, report['confusion'])), count)
                self.assertTrue((output / 'evaluation.png').is_file())
                self.assertEqual(checkpoint.read_bytes(), original)

    def test_perfect_predictions_render_without_mistakes(self):
        images = torch.zeros(1, 1, 28, 28)
        report, mistakes = assess(KnownPredictions(), [(images, torch.tensor([0]))], 'cpu')
        report.update(scope='Test subset', checkpoint='model.pt')
        fig = render_report(report, mistakes)
        try:
            fig.canvas.draw()
            self.assertEqual(mistakes, [])
            self.assertTrue(any('No mistakes' in text.get_text() for text in fig.texts))
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
