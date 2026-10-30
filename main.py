"""Compatibility entry point: python main.py train|predict [options]."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'predict'])
    args, remaining = parser.parse_known_args()
    if args.mode == 'train':
        from train import main
    else:
        from predict import main
    main(remaining)
