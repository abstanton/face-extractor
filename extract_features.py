import numpy as np
from PIL import Image
from hsemotion.facial_emotions import HSEmotionRecognizer
from glob import glob
from argparse import ArgumentParser
import pickle
import os
import tqdm


def splitToBatches(arr, batch_size=10):
    output = []
    curr_batch = []
    for vid in arr:
        if len(curr_batch) < batch_size:
            curr_batch.append(vid)
            continue
        else:
            output.append(curr_batch)
            curr_batch = [vid]
    if curr_batch != []:
        output.append(curr_batch)
    return output


def getFiles(dir):
    dirFiles = os.listdir(dir)
    files = list()
    for file in dirFiles:
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            files = files + getFiles(path)
        else:
            files.append(path)

    return files


def outputFileName(file, input_dir, output_dir):
    # replace top level dir with output dir
    output_file = file.replace(input_dir, output_dir)
    output_file = output_file.replace("jpg", "pickle")
    return output_file


def main(input_directory, output_directory, batch_size):
    # load all images to extract features from and split into batches
    all_files = getFiles(input_directory)
    picture_files = [file for file in all_files if file.endswith("jpg")]
    batches = splitToBatches(picture_files, batch_size)

    fer = HSEmotionRecognizer(device='cuda')

    print("Found {} images".format(len(picture_files)))

    with tqdm.tqdm(total=len(picture_files)) as pbar:
        for batch in batches:
            # load all images in batch
            batch_images = [np.asarray(Image.open(file)) for file in batch]
            features = fer.extract_multi_features(batch_images)

            for (file, features) in zip(batch, features):
                output_file = outputFileName(
                    file, input_directory, output_directory)
                output_file_dir = os.path.dirname(output_file)
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)
                pickle.dump(features, open(output_file, "wb"))
                pbar.update(1)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", default="input",
                        help="Top level directory of input face pictures")
    parser.add_argument("-o", "--output", default="output",
                        help="Top level directory of output features")
    parser.add_argument("-bs", "--batch_size", default=100,
                        help="Batch size of images to process")

    args = vars(parser.parse_args())
    main(args["input"], args["output"], int(args["batch_size"]))
