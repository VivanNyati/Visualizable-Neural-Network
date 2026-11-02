"""Bounded-cost views of real weights, activations and predictions."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

BG, PANEL, TEXT, MUTED = '#0b1220', '#131e30', '#edf3ff', '#9aacc5'
CYAN, GREEN, RED = '#64d8ee', '#71dfad', '#ff8495'


def theme():
    plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': PANEL,
        'axes.edgecolor': '#30415c', 'text.color': TEXT, 'axes.labelcolor': MUTED,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'font.size': 10,
        'axes.titleweight': 'bold', 'axes.titlesize': 11, 'savefig.facecolor': BG})


def array(tensor):
    return tensor.detach().cpu().numpy()


def mosaic(maps, columns=4, gap=1):
    maps = np.asarray(maps)
    count, height, width = maps.shape
    rows = (count + columns - 1) // columns
    result = np.full((rows * (height+gap)-gap, columns * (width+gap)-gap), np.nan)
    for index, tile in enumerate(maps):
        r, c = divmod(index, columns)
        result[r*(height+gap):r*(height+gap)+height,
               c*(width+gap):c*(width+gap)+width] = tile
    return result


def heat(ax, data, title, signed=False):
    ax.set_title(title, loc='left', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if signed:
        limit = max(float(np.nanmax(np.abs(data))), 1e-8)
        return ax.imshow(data, cmap='RdBu_r', vmin=-limit, vmax=limit,
                         interpolation='nearest', aspect='auto')
    return ax.imshow(data, cmap='magma', vmin=0,
                     vmax=max(float(np.nanmax(data)), 1e-8), interpolation='nearest')


def set_heat(artist, data, signed=False):
    artist.set_data(data)
    maximum = max(float(np.nanmax(np.abs(data))), 1e-8)
    artist.set_clim(-maximum if signed else 0, maximum)


def export(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


class TrainingDashboard:
    def __init__(self, model, live=True):
        theme()
        self.live, self.paused, self.stopped = live, False, False
        self.finished = False
        self.steps, self.losses, self.accuracies = [], [], []
        self.validation_steps, self.validation_losses, self.validation_accuracies = [], [], []
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title('Digit Lab | Training')
        self.fig.suptitle('DIGIT LAB  /  LEARNING IN MOTION', x=.055, ha='left',
                          fontsize=19, fontweight='bold', y=.975)
        self.status = self.fig.text(.055, .915, 'Preparing MNIST...', color=MUTED)
        self.save_status = self.fig.text(.055, .105, '', color=GREEN, fontsize=8)
        self.best_status = self.fig.text(.055, .085, '', color=GREEN, fontsize=8)
        grid = self.fig.add_gridspec(2, 3, left=.055, right=.96, top=.85,
                                    bottom=.16, hspace=.45, wspace=.32)
        loss_ax, acc_ax, weights_ax, feature_ax, matrix_ax, sample_ax = [
            self.fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
        self.loss_line, = loss_ax.plot([], [], '.-', color=CYAN, lw=2, ms=3,
                                      label='Training (epoch average)')
        self.acc_line, = acc_ax.plot([], [], '.-', color=GREEN, lw=2, ms=3,
                                     label='Training (epoch average)')
        self.validation_loss_line, = loss_ax.plot([], [], 'o--', color=RED, ms=5,
                                                  label='Validation (epoch end)')
        self.validation_acc_line, = acc_ax.plot([], [], 'o--', color=RED, ms=5,
                                               label='Validation (epoch end)')
        for ax, title, ylabel in [(loss_ax, '01 / Loss', 'Cross entropy'),
                                   (acc_ax, '02 / Accuracy', 'Correct (%)')]:
            ax.set_title(title, loc='left', pad=10)
            ax.set_xlabel('Optimizer step')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.12)
            ax.legend(loc='best', fontsize=7, facecolor=PANEL, edgecolor='#30415c',
                      labelcolor=TEXT)
        acc_ax.set_ylim(0, 100)
        self.weight = heat(weights_ax, mosaic(array(model.conv1.weight[:, 0])),
                           '03 / Learned 3 x 3 filters', True)
        self.feature = heat(feature_ax, mosaic(np.zeros((8, 28, 28))),
                            '04 / First-layer responses | sample 1')
        self.matrix = heat(matrix_ax, array(model.fc2.weight),
                           '05 / Output weights | 10 digits x 64 units', True)
        self.samples = sample_ax.imshow(np.zeros((56, 112)), cmap='gray', vmin=0, vmax=1)
        sample_ax.set_title('06 / Current examples', loc='left', pad=10)
        sample_ax.set_xticks([])
        sample_ax.set_yticks([])
        self.sample_text = sample_ax.text(0, -.12, '', transform=sample_ax.transAxes,
                                          fontsize=8, color=MUTED, va='top')
        self.fig.text(.055, .045,
            'WEIGHTS  blue = negative / red = positive     |     FEATURES  dark = quiet / bright = active\n'
            'All tiles use real tensors. Feature maps share a scale per panel; weight scales adapt per update.',
            fontsize=9, color=MUTED)
        self.loss_ax, self.acc_ax = loss_ax, acc_ax
        if live:
            self.pause_button = Button(self.fig.add_axes([.73, .9, .10, .045]),
                                       'Pause', color=PANEL, hovercolor='#30415c')
            self.stop_button = Button(self.fig.add_axes([.85, .9, .10, .045]),
                                      'Stop & save', color=PANEL, hovercolor='#30415c')
            self.pause_button.on_clicked(self.toggle_pause)
            self.stop_button.on_clicked(self.stop)
            self.fig.canvas.mpl_connect('close_event', self.window_closed)
            plt.ion()
            plt.show(block=False)

    def toggle_pause(self, _):
        if self.stopped or self.finished:
            return
        self.paused = not self.paused
        self.pause_button.label.set_text('Resume' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()

    def stop(self, _):
        if self.finished:
            plt.close(self.fig)
            return
        self.stopped = True
        self.paused = False
        self.saving()

    def window_closed(self, event):
        if not self.finished:
            self.stop(event)

    def saving(self):
        self.paused = False
        self.status.set_text('Saving model and evaluating final results...')
        if self.live:
            self.pause_button.set_active(False)
            self.pause_button.label.set_color(MUTED)
            self.stop_button.set_active(False)
            self.stop_button.label.set_text('Saving...')
        self.fig.canvas.draw_idle()

    def finish(self, status, checkpoint=None):
        self.finished = True
        self.paused = False
        self.status.set_text(status)
        self.save_status.set_text(f'Model saved to: {checkpoint}' if checkpoint else
                                  'No model saved: stopped before the first training step.')
        if self.live:
            self.pause_button.set_active(False)
            self.pause_button.label.set_color(MUTED)
            self.stop_button.label.set_text('Close')
            self.stop_button.set_active(True)
        self.fig.canvas.draw_idle()

    def pump(self):
        if self.live and plt.fignum_exists(self.fig.number):
            self.fig.canvas.flush_events()
            while self.paused and not self.stopped:
                plt.pause(.05)

    def record_best(self, epoch, metrics, checkpoint):
        self.best_status.set_text(f'Best validation: epoch {epoch}   |   '
                                  f'Accuracy: {metrics["accuracy"]:.2%}   |   {Path(checkpoint).name}')

    def begin_epoch(self, epoch):
        # Break the curve where the cumulative training average resets.
        if self.steps:
            self.steps.append(self.steps[-1])
            self.losses.append(float('nan'))
            self.accuracies.append(float('nan'))

    def validation_started(self, epoch):
        self.status.set_text(f'Epoch {epoch}   |   Evaluating validation set...')
        if self.live:
            self.fig.canvas.draw_idle()
            self.pump()

    def record_validation(self, epoch, step, metrics):
        self.validation_steps.append(step)
        self.validation_losses.append(metrics['loss'])
        self.validation_accuracies.append(100 * metrics['accuracy'])
        self.validation_loss_line.set_data(self.validation_steps, self.validation_losses)
        self.validation_acc_line.set_data(self.validation_steps, self.validation_accuracies)
        for ax in (self.loss_ax, self.acc_ax):
            ax.axvline(step, color=MUTED, ls=':', lw=1, alpha=.5)
            ax.text(step, .98, f'E{epoch}', transform=ax.get_xaxis_transform(),
                    ha='right', va='top', color=MUTED, fontsize=7)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        self.acc_ax.set_xlim(0, max(step * 1.03, 1))
        if not self.stopped:
            self.status.set_text(f'Epoch {epoch} complete   |   Validation loss: '
                                 f'{metrics["loss"]:.4f}   |   Accuracy: {metrics["accuracy"]:.1%}')
        if self.live:
            self.fig.canvas.draw_idle()
            self.pump()

    def update(self, model, images, labels, step, loss, accuracy, status):
        self.steps.append(step)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        # Keep redraw cost bounded during long runs.
        if len(self.steps) > 1000:
            self.steps, self.losses, self.accuracies = [v[::2] for v in
                (self.steps, self.losses, self.accuracies)]
        self.loss_line.set_data(self.steps, self.losses)
        self.acc_line.set_data(self.steps, self.accuracies)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        self.acc_ax.set_xlim(0, max(step * 1.03, 1))
        import torch
        with torch.inference_mode():
            logits, trace = model(images[:8], trace=True)
        set_heat(self.weight, mosaic(array(model.conv1.weight[:, 0])), True)
        set_heat(self.matrix, array(model.fc2.weight), True)
        set_heat(self.feature, mosaic(array(trace['conv1'][0])))
        count = min(len(images), 8)
        tiles = np.zeros((8, 28, 28))
        tiles[:count] = array(images[:count, 0]) * .5 + .5
        self.samples.set_data(mosaic(tiles, 4))
        predictions = logits.argmax(1).tolist()
        truths = labels[:count].tolist()
        lines = [f'{p}/{t}' + (' OK' if p == t else ' X') for p, t in zip(predictions, truths)]
        self.sample_text.set_text('Prediction / label (left to right)\n' +
                                  '   '.join(lines[:4]) + '\n' + '   '.join(lines[4:]))
        self.status.set_text(status)
        if self.live:
            self.fig.canvas.draw_idle()
            self.pump()


class PredictionDashboard:
    def __init__(self, model, image, result, name, choose_image=None, polarity='auto'):
        theme()
        self.model, self.result = model, result
        self.choose_image, self.polarity = choose_image, polarity
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('Digit Lab | Image inspector')
        self.fig.suptitle('DIGIT LAB  /  INSIDE A PREDICTION', x=.055, ha='left',
                          fontsize=19, fontweight='bold', y=.975)
        digit = result['prediction']
        self.header = self.fig.text(.055, .925, f'{name[:45]}   |   Prediction: {digit}   |   '
            f'Softmax score: {result["probabilities"][digit]:.2%}', color=CYAN, fontsize=12)
        grid = self.fig.add_gridspec(2, 4, left=.055, right=.96, top=.85,
                                    bottom=.23, hspace=.48, wspace=.4)
        axes = [self.fig.add_subplot(grid[r, c]) for r in range(2) for c in range(4)]
        self.input = heat(axes[0], array(image[0, 0]) * .5 + .5, '01 / Normalized input')
        self.kernel = heat(axes[1], array(model.conv1.weight[0, 0]), '02 / Filter 0', True)
        self.response = heat(axes[2], result['activations']['conv1'][0], '03 / Its ReLU response')
        self.pooled = heat(axes[3], mosaic(result['activations']['pool2']), '04 / Pooled features | 16 x 7 x 7')
        self.hidden = heat(axes[4], result['activations']['hidden'].reshape(8, 8), '05 / Hidden units | 64')
        self.contributions = heat(axes[5], result['contributions'].reshape(8, 8), '06 / Contributions to chosen digit', True)
        self.bars = axes[6].barh(range(10), result['probabilities'], color=[CYAN if d == digit else
                     '#314b70' for d in range(10)])
        axes[6].set(yticks=range(10), xlim=(0, 1), xlabel='Softmax score')
        axes[6].set_title('07 / Class scores', loc='left', pad=10)
        axes[6].invert_yaxis()
        self.saliency = heat(axes[7], result['saliency'], '08 / Pixel sensitivity | gradient')
        self.details = self.fig.text(.055, .145,
            'FLOW   28 x 28 input  >  8 filters + pool  >  16 filters + pool  >  64 units  >  10 digit scores\n'
            f'Contributions are activation x output weight; their sum + bias ({result["bias"]:+.3f}) '
            f'= chosen logit ({result["logit"]:+.3f}).\n'
            'Sensitivity shows local changes in that logit, not a causal explanation. Softmax scores are not calibrated certainty.',
            color=MUTED, fontsize=9, linespacing=1.8)
        self.slider = Slider(self.fig.add_axes([.18, .065, .4, .025], facecolor=PANEL),
                             'First-layer filter', 0, 7, valinit=0, valstep=1, color=CYAN)
        self.slider.on_changed(self.select_filter)
        self.fig.text(.64, .065, 'Blue / red = signed weights or contributions\n'
                      'Bright = stronger activation or sensitivity', color=MUTED, fontsize=9)
        self.message = self.fig.text(.055, .885, '', color=RED, fontsize=9)
        if choose_image is not None:
            self.open_button = Button(self.fig.add_axes([.83, .90, .13, .04]),
                                      'Open image', color=PANEL, hovercolor='#30415c')
            self.open_button.on_clicked(self.open_image)

    def open_image(self, _):
        from images import prepare_image, explain
        try:
            path = self.choose_image()
            if not path:
                return
            image = prepare_image(path, self.polarity).to(next(self.model.parameters()).device)
            result = explain(self.model, image)
        except (OSError, ValueError, RuntimeError) as error:
            self.message.set_text(f'Could not open image: {str(error)[:140]}')
            self.fig.canvas.draw_idle()
            return
        self.result = result
        digit = result['prediction']
        self.header.set_text(f'{Path(path).name[:45]}   |   Prediction: {digit}   |   '
                             f'Softmax score: {result["probabilities"][digit]:.2%}')
        self.input.set_data(array(image[0, 0]) * .5 + .5)
        set_heat(self.pooled, mosaic(result['activations']['pool2']))
        set_heat(self.hidden, result['activations']['hidden'].reshape(8, 8))
        set_heat(self.contributions, result['contributions'].reshape(8, 8), True)
        set_heat(self.saliency, result['saliency'])
        for index, bar in enumerate(self.bars):
            bar.set_width(result['probabilities'][index])
            bar.set_color(CYAN if index == digit else '#314b70')
        self.details.set_text(
            'FLOW   28 x 28 input  >  8 filters + pool  >  16 filters + pool  >  64 units  >  10 digit scores\n'
            f'Contributions are activation x output weight; their sum + bias ({result["bias"]:+.3f}) '
            f'= chosen logit ({result["logit"]:+.3f}).\n'
            'Sensitivity shows local changes in that logit, not a causal explanation. Softmax scores are not calibrated certainty.')
        self.message.set_text('')
        self.select_filter(self.slider.val)

    def select_filter(self, value):
        index = int(value)
        set_heat(self.kernel, array(self.model.conv1.weight[index, 0]), True)
        set_heat(self.response, self.result['activations']['conv1'][index])
        self.kernel.axes.set_title(f'02 / Filter {index}', loc='left', pad=10)
        self.fig.canvas.draw_idle()
