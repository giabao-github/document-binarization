"""CLI for evaluation."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate binarization results")
    parser.add_argument("pred", help="Path to predicted binary image")
    parser.add_argument("gt", help="Path to ground truth image")
    args = parser.parse_args()
    print(f"Would evaluate {args.pred} vs {args.gt}")


if __name__ == "__main__":
    main()
