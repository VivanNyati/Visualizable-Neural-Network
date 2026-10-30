"""Run with: python -m unittest -v test_core."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from model import DigitNet, save_checkpoint, load_checkpoint
from images import prepare_image, explain


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)

    def test_trace_matches_forward_and_contributions_reconstruct_logit(self):
        model = DigitNet().eval()
        x = torch.randn(1, 1, 28, 28)
        logits, trace = model(x, trace=True)
        torch.testing.assert_close(logits, model(x))
        self.assertEqual(trace['pool2'].shape, (1, 16, 7, 7))
        result = explain(model, x)
        self.assertAlmostEqual(float(result['probabilities'].sum()), 1., places=6)
        self.assertAlmostEqual(float(result['contributions'].sum()) + result['bias'],
                               result['logit'], places=5)
        self.assertTrue(np.isfinite(result['saliency']).all())
        self.assertGreater(float(result['saliency'].max()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_training_changes_weights_and_checkpoint_round_trip(self):
        model = DigitNet()
        x, labels = torch.randn(4, 1, 28, 28), torch.tensor([1, 2, 3, 4])
        before = model.conv1.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters())
        torch.nn.functional.cross_entropy(model(x), labels).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.conv1.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(model, path, {'steps': 1})
            restored, metadata = load_checkpoint(path, 'cpu')
            torch.testing.assert_close(model(x), restored(x))
            self.assertEqual(metadata['steps'], 1)

    def test_polarity_transparency_centering_and_blank_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'digit.png'
            image = Image.new('L', (100, 80), 'white')
            ImageDraw.Draw(image).line([(12, 10), (40, 10), (20, 60)], fill=0, width=7)
            image.save(path)
            dark = prepare_image(path)
            ImageOps.invert(image).save(path)
            light = prepare_image(path)
            torch.testing.assert_close(dark, light)
            self.assertEqual(dark.shape, (1, 1, 28, 28))
            self.assertGreaterEqual(float(dark.min()), -1)
            self.assertLessEqual(float(dark.max()), 1)
            ink = (dark[0, 0].numpy() + 1) / 2
            yy, xx = np.indices(ink.shape)
            self.assertLess(abs(float((xx*ink).sum()/ink.sum()) - 13.5), .6)
            rgba = Image.new('RGBA', image.size, (255, 255, 255, 0))
            ImageDraw.Draw(rgba).line([(12, 10), (40, 10), (20, 60)], fill='black', width=7)
            rgba.save(path)
            torch.testing.assert_close(dark, prepare_image(path))
            Image.new('L', (28, 28), 'white').save(path)
            with self.assertRaisesRegex(ValueError, 'blank'):
                prepare_image(path)


if __name__ == '__main__':
    unittest.main()
