import os
import argparse
import logging
from glob import glob
from PIL import Image

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output_data", type=str, help="path to output data")
    args = parser.parse_args()


    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    print("output folder:", args.output_data)

    output_dir = args.output_data
    size = (64, 64) # Later we can also pass this as a property


    for file in glob(args.data + "/*.jpg"):
        img = Image.open(file)
        img_resized = img.resize(size)

        # Save the resized image to the output directory
        output_file = os.path.join(output_dir, os.path.basename(file))
        img_resized.save(output_file)


if __name__ == "__main__":
    main()