"""Headless rendering and interaction checks: python -m unittest -v test_dashboard."""
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from dashboard import TrainingDashboard, PredictionDashboard
from images import explain
from model import DigitNet


class DashboardTests(unittest.TestCase):
    def test_training_handles_small_batch_and_reuses_axes(self):
        torch.set_num_threads(2)
        model = DigitNet()
        view = TrainingDashboard(model, live=False)
        axes = len(view.fig.axes)
        try:
            for count in [8, 1]:
                view.update(model, torch.randn(count, 1, 28, 28),
                            torch.zeros(count, dtype=torch.long), count, .5, 50., 'Test')
                view.fig.canvas.draw()
                self.assertEqual(len(view.fig.axes), axes)
            view.stop(None)
            self.assertTrue(view.stopped)
        finally:
            plt.close(view.fig)

    def test_filter_slider_switches_to_actual_kernel(self):
        torch.set_num_threads(2)
        model = DigitNet().eval()
        image = torch.randn(1, 1, 28, 28)
        view = PredictionDashboard(model, image, explain(model, image), 'test.png')
        try:
            view.slider.set_val(7)
            torch.testing.assert_close(torch.tensor(view.kernel.get_array().copy()),
                                       model.conv1.weight[7, 0].detach())
            view.fig.canvas.draw()
        finally:
            plt.close(view.fig)


if __name__ == '__main__':
    unittest.main()
