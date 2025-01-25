import cv2
import numpy as np


def image_pinjie(img1_path, img2_path):
    MIN = 10
    FLANN_INDEX_KDTREE = 0

    img1 = cv2.imread(img1_path)  # query
    img2 = cv2.imread(img2_path)  # train
    imageA = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2)
    imageB = cv2.resize(img2, (0, 0), fx=0.2, fy=0.2)

    sift = cv2.SIFT_create()
    kp1, descrip1 = sift.detectAndCompute(imageA, None)
    kp2, descrip2 = sift.detectAndCompute(imageB, None)

    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    match = flann.knnMatch(descrip1, descrip2, k=2)
    good = []

    for i, (m, n) in enumerate(match):
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > MIN:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
        warpImg = cv2.warpPerspective(imageB, np.linalg.inv(M), (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
        direct = warpImg.copy()
        direct[0:imageA.shape[0], 0:imageB.shape[1]] = imageA

    rows, cols = imageA.shape[:2]
    for col in range(0, cols):
        if imageA[:, col].any() and warpImg[:, col].any():
            left = col
            break

    for col in range(cols - 1, 0, -1):
        if imageA[:, col].any() and warpImg[:, col].any():
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not imageA[row, col].any():
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = imageA[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(imageA[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    warpImg[0:imageA.shape[0], 0:imageA.shape[1]] = res

    return warpImg


# # 测试函数
# if __name__ == "__main__":
#     img1_path = "C:\\Users\\86155\\Desktop\\zuo.jpg"
#     img2_path = "C:\\Users\\86155\\Desktop\\you.jpg"
#     stitched_image = image_pinjie(img1_path, img2_path)
#     cv2.imshow('Stitched Image', stitched_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
