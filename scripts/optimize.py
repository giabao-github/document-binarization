"""CLI for optimization experiments."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Optimize binarization parameters")
    parser.add_argument("config", help="Config file or preset")
    args = parser.parse_args()
    print(f"Would optimize using {args.config}")


if __name__ == "__main__":
    main()
