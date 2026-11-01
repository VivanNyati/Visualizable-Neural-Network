"""Run preservation and prediction selection checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model import ROOT
from runs import create_run, list_runs, select_checkpoint
from test_train import TinyDigits
from train import main as train
from predict import main as predict


class RunTests(unittest.TestCase):
    def test_two_default_runs_preserve_models_and_support_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('train.create_run', side_effect=lambda path: create_run(path, root)), \
                    patch('torchvision.datasets.MNIST', return_value=TinyDigits()):
                arguments = ['--no-ui', '--epochs', '1', '--max-train-samples', '64',
                             '--max-eval-samples', '16']
                train(arguments)
                first = list_runs(root)[0]
                original = Path(first['checkpoint']).read_bytes()
                train(arguments)
            available = list_runs(root)
            self.assertEqual(len(available), 2)
            self.assertEqual(Path(first['checkpoint']).read_bytes(), original)
            self.assertNotEqual(available[0]['name'], first['name'])
            self.assertEqual(select_checkpoint(root=root), Path(available[0]['checkpoint']))
            self.assertEqual(select_checkpoint(first['name'], 'best', root),
                             Path(first['best_checkpoint']))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                predict(['--list-runs', '--runs-dir', str(root)])
            self.assertIn(first['name'], output.getvalue())
            predict([str(ROOT / 'examples' / 'digit.png'), '--runs-dir', str(root),
                     '--run', first['name'], '--weights', 'best', '--no-ui',
                     '--output', str(root / 'prediction.png')])
            self.assertTrue((root / 'prediction.png').is_file())
            with self.assertRaises(ValueError):
                create_run(root / first['name'])

    def test_missing_and_incomplete_runs_are_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = create_run(root / 'incomplete')
            (folder / 'run.json').write_text('{invalid', encoding='utf-8')
            self.assertEqual(list_runs(root), [])
            with self.assertRaises(ValueError):
                select_checkpoint(root=root)
            checkpoint = folder / 'model.pt'
            checkpoint.write_bytes(b'placeholder for listing only')
            (folder / 'run.json').write_text(json.dumps({
                'created': '2026-01-01', 'checkpoint': str(checkpoint),
                'epochs_completed': 0, 'interrupted': True}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'no best checkpoint'):
                select_checkpoint('incomplete', 'best', root)
            checkpoint.unlink()
            self.assertEqual(list_runs(root), [])


if __name__ == '__main__':
    unittest.main()
