"""CLI for running binarization pipelines."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Binarize an input image")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path for output binary image")
    parser.add_argument("--method", default="otsu", help="Binarization method")
    args = parser.parse_args()
    print(f"Would binarize {args.input} -> {args.output} using {args.method}")


if __name__ == "__main__":
    main()
