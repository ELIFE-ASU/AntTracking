import os
import cv2
import numpy as np

SOURCE = '../data/ant_img'
TARGET = '../data/ant_img_gs'
SIZE = 28
AUG = 5


def main():
    for sub in os.listdir(SOURCE):
        sub_folder = os.path.join(SOURCE, sub)
        if not sub.startswith('.') and os.path.isdir(sub_folder):
            target_sub_folder = os.path.join(TARGET, sub)
            if not os.path.exists(target_sub_folder):
                os.mkdir(target_sub_folder)
            for item in os.listdir(sub_folder):
                afile = os.path.join(sub_folder, item)
                if not item.startswith('.') and os.path.isfile(afile):
                    # Read image as gray_scale
                    img = cv2.imread(afile, 0)
                    _, th = cv2.threshold(img, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Subtract background
                    mask = th == 0
                    masked = mask * img
                    # Find blob centroids
                    _, contours, _ = cv2.findContours(th, 1, 2)
                    M = list(map(cv2.moments, contours))
                    try:
                        cxs = list(map(lambda x: int(x['m10'] / x['m00']), M))
                        cys = list(map(lambda x: int(x['m01'] / x['m00']), M))
                    except ZeroDivisionError:
                        print('Error finding centroids for {}/ {}'.format(sub, item))
                        continue
                    # Find center of all blob centroids
                    cx = sum(cxs) // len(cxs)
                    cy = sum(cys) // len(cys)
                    # Crop image near the center
                    if cx < SIZE // 2 or cy < SIZE // 2:
                        print('{}/{} centoirds off center'.format(sub, item))
                        continue
                    cropped = masked[cy - SIZE // 2:cy +
                                     SIZE // 2, cx - SIZE // 2:cx + SIZE // 2]
                    # Save cropped image
                    cv2.imwrite(os.path.join(target_sub_folder, item), cropped)
                    # Rotation
                    rows, cols = cropped.shape
                    for i in range(AUG):
                        mat = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                                      np.random.rand() * 360, 1)
                        rotated = cv2.warpAffine(cropped, mat, (cols, rows))
                        # New file name
                        mod_item = item[:-4] + '_rot' + str(i) + '.png'
                        # Save rotated image
                        cv2.imwrite(os.path.join(TARGET, sub, mod_item),
                                    rotated)


if __name__ == '__main__':
    main()
