import random
import timeit
import cv2
import numpy as np


def graph_based_segmentation(image, sigma=0.5, k=300, min_size=1000):
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=sigma, k=k, min_size=min_size)

    start = timeit.default_timer()
    segments = segmentator.processImage(image)
    print('segments process time: {:.2f} sec'.format(timeit.default_timer() - start))
    print('total segments count：{}'.format(np.max(segments)))

    seg_image = np.zeros(image.shape, np.uint8)

    for i in range(np.max(segments)):
        y, x = np.where(segments == i)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for xi, yi in zip(x, y):
            seg_image[yi, xi] = color

    image = cv2.addWeighted(image, 0.3, seg_image, 0.7, 0)

    return image


def selective_search(image):
    selection_maker = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selection_maker.setBaseImage(image)

    # 使用快速模式（精準度較差）
    selection_maker.switchToSelectiveSearchFast()
    # 使用精準模式（速度較慢）
    # ss.switchToSelectiveSearchQuality()

    start = timeit.default_timer()
    rects = selection_maker.process()
    print('selective process time: {:.2f} sec'.format(timeit.default_timer() - start))
    print('total region count：{}'.format(len(rects)))

    num_to_show = 100

    for rect in rects[:num_to_show]:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    return image


def main():
    img = cv2.imread('images/taipei.jpg')
    seg = graph_based_segmentation(img, sigma=0.5, k=300, min_size=1000)
    ss = selective_search(img)

    cv2.imshow("segmentation", seg)
    cv2.imshow("selective", ss)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
