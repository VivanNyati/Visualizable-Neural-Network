"""Headless rendering and interaction checks: python -m unittest -v test_dashboard."""
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import torch
from dashboard import TrainingDashboard, PredictionDashboard
from images import explain
from model import DigitNet


class DashboardTests(unittest.TestCase):
    def test_validation_points_epoch_gaps_and_loss_scale(self):
        model = DigitNet()
        view = TrainingDashboard(model, live=False)
        try:
            images = torch.randn(1, 1, 28, 28)
            labels = torch.zeros(1, dtype=torch.long)
            view.begin_epoch(1)
            view.update(model, images, labels, 10, .5, 80, 'Epoch 1')
            view.validation_started(1)
            self.assertIn('Evaluating validation', view.status.get_text())
            view.record_validation(1, 10, {'loss': 3., 'accuracy': .75})
            view.begin_epoch(2)
            view.update(model, images, labels, 20, .2, 90, 'Epoch 2')
            view.record_validation(2, 20, {'loss': 2., 'accuracy': .85})
            self.assertTrue(torch.isnan(torch.tensor(view.loss_line.get_ydata()[1])))
            self.assertEqual(list(view.validation_loss_line.get_xdata()), [10, 20])
            self.assertEqual(list(view.validation_acc_line.get_ydata()), [75, 85])
            self.assertGreater(view.loss_ax.get_ylim()[1], 3.)
            self.assertEqual([text.get_text() for text in view.loss_ax.texts], ['E1', 'E2'])
            self.assertEqual(view.acc_ax.get_ylim(), (0, 100))
            view.fig.canvas.draw()
        finally:
            plt.close(view.fig)

    def click(self, view, button):
        view.fig.canvas.draw()
        x, y = button.ax.transAxes.transform((.5, .5))
        for name in ['button_press_event', 'button_release_event']:
            event = MouseEvent(name, view.fig.canvas, x, y, button=1)
            view.fig.canvas.callbacks.process(name, event)

    def test_stop_while_paused_then_close_saved_results(self):
        with patch('matplotlib.pyplot.show'):
            view = TrainingDashboard(DigitNet(), live=True)
        try:
            self.click(view, view.pause_button)
            self.assertTrue(view.paused)
            self.click(view, view.stop_button)
            self.assertTrue(view.stopped)
            self.assertFalse(view.paused)
            self.assertFalse(view.pause_button.active)
            self.assertEqual(view.stop_button.label.get_text(), 'Saving...')
            view.finish('Stopped & saved', 'runs/preview/mnist.pt')
            self.assertIn('runs/preview/mnist.pt', view.save_status.get_text())
            self.assertEqual(view.stop_button.label.get_text(), 'Close')
            self.assertTrue(plt.fignum_exists(view.fig.number))
            self.click(view, view.stop_button)
            self.assertFalse(plt.fignum_exists(view.fig.number))
        finally:
            plt.close(view.fig)
            plt.ioff()

    def test_natural_completion_disables_pause_and_close_works(self):
        with patch('matplotlib.pyplot.show'):
            view = TrainingDashboard(DigitNet(), live=True)
        try:
            view.saving()
            view.finish('Finished & saved', 'mnist.pt')
            self.click(view, view.pause_button)
            self.assertFalse(view.paused)
            self.assertFalse(view.pause_button.active)
            self.click(view, view.stop_button)
            self.assertFalse(plt.fignum_exists(view.fig.number))
        finally:
            plt.close(view.fig)
            plt.ioff()

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
