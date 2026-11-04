"""Compatibility entry point: python main.py train|predict|evaluate [options]."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'predict', 'evaluate'])
    args, remaining = parser.parse_known_args()
    if args.mode == 'train':
        from train import main
    elif args.mode == 'predict':
        from predict import main
    else:
        from evaluate import main
    main(remaining)
