# from hand_tracking_module import HandDetector
import hand_tracking_module as htm
import mediapipe as mp
import cv2
import os
import math


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def resize(img, DESIRED_HEIGHT=480, DESIRED_WIDTH=480):
    h, w = img.shape[:2]
    if h < w:
        return cv2.resize(img, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        return cv2.resize(img, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))


def main():
    images = load_images_from_folder('./hands')
    detector = htm.HandDetector(mode=True)

    print(detector)

    id = 0
    for img in images:
        cv2.imshow(f"Image {id}", detector.findHands(resize(img)))
        id += 1

    index = 2
    cv2.imshow(f"Image {id}", detector.findHands(resize(images[index])))

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
