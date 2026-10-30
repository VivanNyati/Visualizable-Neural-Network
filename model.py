"""Small, explicitly traceable MNIST CNN. No persistent activation hooks."""
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'mnist.pt'


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, trace=False):
        first = torch.relu(self.conv1(x))
        pool1 = torch.nn.functional.max_pool2d(first, 2)
        second = torch.relu(self.conv2(pool1))
        pool2 = torch.nn.functional.max_pool2d(second, 2)
        hidden = torch.relu(self.fc1(pool2.flatten(1)))
        logits = self.fc2(hidden)
        if trace:
            return logits, {'conv1': first, 'pool1': pool1, 'conv2': second,
                            'pool2': pool2, 'hidden': hidden}
        return logits


def get_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')


def save_checkpoint(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'format_version': 1, 'state_dict': {k: v.detach().cpu() for k, v
                in model.state_dict().items()}, 'metadata': metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint; train with this version first.')
    model = DigitNet()
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint['metadata']
