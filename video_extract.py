# USAGE
# python extract.py --output outputDir/ --input someDir/
# python extract.py --output outputDir/ --input image.png
# python extract.py --output outputDir/ --input video.mp4

import os
import cv2
import argparse
from PIL import Image
from retinaface import RetinaFace
from tqdm import tqdm

VIDEO_TEMP_IMAGE_PATH = "temp.jpg"


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


''' get ratio of dist between eyes to width of face '''


def getEyeDistRatio(face_det):
    return abs((face_det['landmarks']['left_eye'][0] - face_det['landmarks']['right_eye'][0]) /
               (face_det['facial_area'][2] - face_det['facial_area'][0]))


def splitToBatches(video_list, batch_size=10):
    output = []
    curr_batch = []
    for vid in video_list:
        if len(curr_batch) < batch_size:
            curr_batch.append(vid)
            continue
        else:
            output.append(curr_batch)
            curr_batch = [vid]
    if curr_batch != []:
        output.append(curr_batch)
    return output


def main(args):
    input = args["input"]
    scale = float(args["scale"])
    size_limit = int(args["size_limit"])
    threshold = float(args["threshold"])
    eye_ratio = float(args["eye_ratio"])
    video_batch_size = int(args["video_batch_size"])

    isDirectory = os.path.isdir(input)
    sources = []
    if isDirectory:
        sources.extend(getFiles(input))
    else:
        sources.append(input)

    total = 0
    cwd = os.getcwd()

    print("Found {} videos".format(len(sources)))
    batches = splitToBatches(sources, video_batch_size)
    with tqdm(total=len(batches)) as batch_bar:
        with tqdm(total=0) as image_bar:
            for batch in batches:
                images = []
                for path in batch:
                    if not path.endswith("mp4"):
                        continue

                    filename = os.path.splitext(os.path.basename(path))[0]
                    outputPath = path.replace(args["input"], args["output"])

                    rate = int(args["rate"])

                    try:
                        video = cv2.VideoCapture(path)
                    except:
                        continue

                    curr_gap = rate+1
                    while True:
                        success, frame = video.read()
                        if curr_gap < rate:
                            curr_gap += 1
                            continue

                        curr_gap = 0

                        if success:
                            image = {
                                "file": frame,
                                "source": VIDEO_TEMP_IMAGE_PATH,
                                "sourceType": "video",
                                "outputPath": outputPath,
                                "filename": filename
                            }
                            images.append(image)
                        else:
                            break
                    video.release()

                image_bar.reset(total=len(images))
                for (i, image) in enumerate(images):
                    image_bar.update(1)
                    # if type is video, save to temp
                    if image["sourceType"] == "video":
                        cv2.imwrite(image["source"], image["file"])

                    results = RetinaFace.detect_faces(image["source"])

                    array = cv2.cvtColor(image['file'], cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(array)

                    if type(results) != dict:
                        continue

                    j = 1
                    for key, face in results.items():
                        if face["score"] < threshold:
                            continue

                        if getEyeDistRatio(face) < eye_ratio:
                            continue

                        (startX, startY, endX, endY) = face['facial_area']

                        bW = endX - startX
                        bH = endY - startY

                        centerX = startX + (bW / 2.0)
                        centerY = startY + (bH / 2.0)
                        left = centerX - bW / 2.0 * scale
                        top = centerY - bH / 2.0 * scale
                        right = centerX + bW / 2.0 * scale
                        bottom = centerY + bH / 2.0 * scale
                        face = img.crop((left, top, right, bottom))
                        fW, fH = face.size

                        if fW < size_limit or fH < size_limit:
                            continue

                        outputFilename = ''
                        if image["sourceType"] == "video":
                            outputFilename = '{}_{:04d}_{}.jpg'.format(
                                image["filename"], i, j)
                        else:
                            outputFilename = '{}_{}.jpg'.format(
                                image["filename"], j)

                        outputDir = os.path.dirname(
                            os.path.join(cwd, image["outputPath"]))
                        if not os.path.exists(outputDir):
                            os.makedirs(outputDir)
                        outputPath = os.path.join(outputDir, outputFilename)
                        face.save(outputPath)
                        total += 1
                        j += 1
                batch_bar.update(1)

    if os.path.exists(VIDEO_TEMP_IMAGE_PATH):
        os.remove(VIDEO_TEMP_IMAGE_PATH)

    print("[INFO] found {} face(s)".format(total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument("-i", "--input", required=True,
                        help="path to tpp level input directory")
    parser.add_argument("-o", "--output", default="output/",
                        help="path to output directory of faces")
    parser.add_argument("-s", "--scale", default=1.0,
                        help="scale of detection area (default: 1)")
    parser.add_argument("-r", "--rate", default=10,
                        help="number of frames between image captures on video")
    parser.add_argument("-sl", "--size_limit", default=0,
                        help="Size limit on either axis for face to be saved")
    parser.add_argument("-t", "--threshold", default=0.9,
                        help="Confidence threshold for face extraction")
    parser.add_argument("-er", "--eye_ratio", default=0.30,
                        help="Threshold for ratio of dist between eyes to width of face, to filter out side profile faces")
    parser.add_argument("-vb", "--video_batch_size", default=100,
                        help="Number of films to process storing in memory")

    args = vars(parser.parse_args())
    main(args)
