"""Sample a random subset from a JSONL file."""
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    random.seed(args.seed)
    sampled = random.sample(lines, min(args.n, len(lines)))

    with open(args.output, 'w') as f:
        f.writelines(sampled)

    print(f'Sampled {len(sampled)} / {len(lines)} lines -> {args.output}')


if __name__ == '__main__':
    main()
