"""Train MNIST: python train.py [--epochs 4] [--no-ui]."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from model import DigitNet, ROOT, DEFAULT_CHECKPOINT, get_device, save_checkpoint


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0., 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return {'loss': loss / count, 'accuracy': correct / count, 'samples': count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=positive, default=4)
    parser.add_argument('--batch-size', type=positive, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--threads', type=positive, default=2, help='CPU threads; limits resource use')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / 'latest')
    parser.add_argument('--max-train-samples', type=positive)
    parser.add_argument('--max-eval-samples', type=positive)
    parser.add_argument('--viz-seconds', type=float, default=1., help='Minimum seconds between redraws')
    parser.add_argument('--no-ui', action='store_true', help='Train without a GUI; save dashboard PNG')
    args = parser.parse_args(argv)
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not np.isfinite(args.viz_seconds) or args.viz_seconds < .1:
        parser.error('--viz-seconds must be finite and at least 0.1')
    if args.no_ui:
        import matplotlib
        matplotlib.use('Agg')
    from dashboard import TrainingDashboard, export
    import matplotlib.pyplot as plt

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
    print('Loading MNIST (first run downloads the dataset)...', flush=True)
    full = datasets.MNIST(args.data_dir, train=True, transform=transform, download=True)
    train, validation = random_split(full, [55000, 5000],
                                     generator=torch.Generator().manual_seed(args.seed))
    test = datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
    if args.max_train_samples:
        train = Subset(train, range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        validation = Subset(validation, range(min(args.max_eval_samples, len(validation))))
        test = Subset(test, range(min(args.max_eval_samples, len(test))))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=256, num_workers=0)
    test_loader = DataLoader(test, batch_size=256, num_workers=0)
    model = DigitNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dashboard = TrainingDashboard(model, live=not args.no_ui)
    print(f'{device} | {sum(p.numel() for p in model.parameters()):,} parameters | '
          f'{len(train):,} training samples', flush=True)
    history, step, epochs_completed = [], 0, 0
    last_draw, started = 0., time.monotonic()
    interrupted = False
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(args.epochs):
            model.train()
            total_loss, correct, count = 0., 0, 0
            for batch, (images, labels) in enumerate(train_loader, 1):
                dashboard.pump()
                if dashboard.stopped:
                    interrupted = True
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                step += 1
                count += len(labels)
                total_loss += loss.item() * len(labels)
                correct += (logits.detach().argmax(1) == labels).sum().item()
                now = time.monotonic()
                if now - last_draw >= args.viz_seconds or batch == len(train_loader):
                    status = (f'Epoch {epoch+1}/{args.epochs}   |   Batch {batch}/{len(train_loader)}'
                              f'   |   {device}   |   Loss {total_loss/count:.4f}'
                              f'   |   Accuracy {correct/count:.1%}')
                    dashboard.update(model, images, labels, step, total_loss/count,
                                     100*correct/count, status)
                    last_draw = now
            if interrupted:
                break
            validation_metrics = evaluate(model, val_loader, device)
            epochs_completed = epoch + 1
            record = {'epoch': epochs_completed, 'train_loss': total_loss/count,
                      'train_accuracy': correct/count, 'validation': validation_metrics}
            history.append(record)
            print(f'Epoch {epoch+1}: loss={total_loss/count:.4f}, '
                  f'validation accuracy={validation_metrics["accuracy"]:.2%}', flush=True)
            # A usable checkpoint survives even if a subsequent epoch is interrupted.
            save_checkpoint(model, args.checkpoint, {'epochs_completed': epochs_completed,
                            'steps': step, 'seed': args.seed, 'history': history})
    except KeyboardInterrupt:
        interrupted = True
        print('\nInterrupted; saving current weights.', flush=True)

    test_metrics = evaluate(model, test_loader, device) if step else None
    metadata = {'epochs_completed': epochs_completed, 'steps': step, 'seed': args.seed,
                'interrupted': interrupted, 'history': history, 'test': test_metrics,
                'train_samples': len(train), 'elapsed_seconds': time.monotonic()-started}
    if step:
        save_checkpoint(model, args.checkpoint, metadata)
    (args.run_dir / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    status = 'Stopped' if interrupted else 'Finished'
    suffix = f'Test accuracy: {test_metrics["accuracy"]:.2%}' if test_metrics else 'No training steps'
    dashboard.status.set_text(f'{status}   |   {suffix}   |   {step} optimizer steps')
    export(dashboard.fig, args.run_dir / 'training.png')
    print(f'{status}. {suffix}. Checkpoint: {args.checkpoint if step else "not saved"}', flush=True)
    if not args.no_ui and not dashboard.stopped:
        plt.ioff()
        plt.show()
    plt.close(dashboard.fig)


if __name__ == '__main__':
    main()
