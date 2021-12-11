import face_alignment
from skimage import io
import cv2
import numpy as np

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('data/rendered.png')
input = input[..., :3]
preds = fa.get_landmarks(input)

input = np.array(input)
print(input.shape)
input = input[..., ::-1].astype(np.uint8).copy()
print(input.shape, input.dtype)
for p in preds[0]:
    p = (int(p[0]), int(p[1]))
    cv2.circle(input, p, 1, [0, 0, 255], -1)

cv2.imwrite('data/detected.png', input)

with open('data/detected.txt', 'w') as f:
    for p in preds[0]:
        f.write(str(p[0]) + "," + str(p[1])+"\n")
