"""Run experiments end-to-end (stub)."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument("manifest", help="Dataset manifest file")
    args = parser.parse_args()
    print(f"Would run experiment for {args.manifest}")


if __name__ == "__main__":
    main()
