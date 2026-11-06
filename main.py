"""Compatibility entry point: python main.py train|predict|evaluate|doctor [options]."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'predict', 'evaluate', 'doctor'])
    args, remaining = parser.parse_known_args()
    if args.mode == 'train':
        from train import main
    elif args.mode == 'predict':
        from predict import main
    elif args.mode == 'evaluate':
        from evaluate import main
    else:
        from doctor import main
    raise SystemExit(main(remaining))
