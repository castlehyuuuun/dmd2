import random
import numpy as np
import cv2


def addWhiteNoise(image):
    # 무작위 확률에 대한 범위 설정
    # 높은 확률은 더 많은 노이즈를 의미합니다.
    prob = random.uniform(0.02, 0.5)

    # 입력 이미지의 모양과 같은 형태의 무작위 행렬 생성
    rnd = np.random.rand(image.shape[0], image.shape[1])
    mask = rnd < prob

    noise = np.random.randint(0, 256, size=(image.shape[0], image.shape[1], 3), dtype=np.uint8)
    image[mask] = noise[mask]
    # rnd 행렬의 무작위 값이 무작위 확률보다 작으면
    # 입력 이미지의 해당 픽셀을 지정된 범위 내의 값으로 무작위로 변경합니다.
    # image[rnd < prob] = np.random.randint(50,230)

    return image

# Load our image
image = cv2.imread("C:/Users/user/Pictures/suri.png")
# cv2.imshow("Input Image", image)

# Apply our white noise function to our input image
noise_1 = addWhiteNoise(image)
cv2.imshow("Noise Added", noise_1)
# cv2.waitKey(-1)
# cv2.fastNlMeansDenoisingColored(input, None, h, hForColorComponents, templateWindowSize, searchWindowSize)
# None are - the filter strength 'h' (5-12 is a good range)
# Next is hForColorComponents, set as same value as h again
# templateWindowSize (odd numbers only) rec. 7
# searchWindowSize (odd numbers only) rec. 21

dst = cv2.fastNlMeansDenoisingColored(noise_1, None, 11, 6, 7, 21)

cv2.imshow("Noise Removed", dst)
cv2.waitKey(-1)