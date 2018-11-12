import cv2
import numpy as np
from math import sqrt as sqrt
import heapq

# parameters
in_path = 'test.png'
drawsize = 5

# global variables
INF = 1e7
MIN = 1e-7
BAND = 1
KNOWN = 0
UNKNOWN = -1


def solve(x1, y1, x2, y2, h, w, dists, flags):
    if y1 < 0 or y1 >= h or x1 < 0 or x1 >= w:
        return INF

    if y2 < 0 or y2 >= h or x2 < 0 or x2 >= w:
        return INF

    flag1 = flags[x1, y1]
    flag2 = flags[x2, y2]

    if flag1 == KNOWN and flag2 == KNOWN:
        d1 = dists[x1, y1]
        d2 = dists[x2, y2]
        d = 2.0 - (d1 - d2) ** 2
        if d > 0.0:
            r = sqrt(d)
            s = (d1 + d2 - r) / 2.0
            if s >= d1 and s >= d2:
                return s
            elif s + r >= d1 and s + r >= d2:
                return s + r
            return INF
    if flag1 == KNOWN:
        return 1.0 + dists[x1, y1]
    if flag2 == KNOWN:
        return 1.0 + dists[x2, y2]
    return INF


def inpaint(src_img, mask, radius=5):
    height, width = src_img.shape[0:2]

    dists = np.full((width, height), INF, dtype=float)
    flags = mask.astype(int) * UNKNOWN
    band = []

    mask_x, mask_y = mask.nonzero()
    for x, y in zip(mask_x, mask_y):
        neighbors = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y)]
        for nx, ny in neighbors:
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if flags[nx, ny] == BAND:
                continue
            if mask[nx, ny] == 0:
                flags[nx, ny] = BAND
                dists[nx, ny] = 0.0
                heapq.heappush(band, (0.0, nx, ny))

    while band:
        _, x, y = heapq.heappop(band)
        flags[x, y] = KNOWN

        neighbors = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y)]
        for nx, ny in neighbors:
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue

            if flags[nx, ny] != UNKNOWN:
                continue

            dist = min([
                solve(nx, ny - 1, nx - 1, ny, height, width, dists, flags),
                solve(nx, ny + 1, nx + 1, ny, height, width, dists, flags),
                solve(nx, ny - 1, nx + 1, ny, height, width, dists, flags),
                solve(nx, ny + 1, nx - 1, ny, height, width, dists, flags)
            ])

            dists[nx, ny] = dist

            grad_x = INF
            prev_x = nx - 1
            next_x = nx + 1
            if prev_x >= 0 and next_x < width:
                flag_prev_x = flags[prev_x, ny]
                flag_next_x = flags[next_x, ny]

                if flag_prev_x != UNKNOWN and flag_next_x != UNKNOWN:
                    grad_x = (dists[next_x, ny] - dists[prev_x, ny]) / 2.0
                elif flag_prev_x != UNKNOWN:
                    grad_x = dist - dists[prev_x, ny]
                elif flag_next_x != UNKNOWN:
                    grad_x = dists[next_x, ny] - dist
                else:
                    grad_x = 0.0

            grad_y = INF
            prev_y = ny - 1
            next_y = ny + 1
            if prev_y >= 0 and next_y < height:
                flag_prev_y = flags[nx, prev_y]
                flag_next_y = flags[nx, next_y]

                if flag_prev_y != UNKNOWN and flag_next_y != UNKNOWN:
                    grad_y = (dists[nx, next_y] - dists[nx, prev_y]) / 2.0
                elif flag_prev_y != UNKNOWN:
                    grad_y = dist - dists[nx, prev_y]
                elif flag_next_y != UNKNOWN:
                    grad_y = dists[nx, next_y] - dist
                else:
                    grad_y = 0.0

            pixel_sum = np.zeros(3, dtype=float)
            weight_sum = 0.0

            for nb1_y in range(ny - radius, ny + radius + 1):
                if nb1_y < 0 or nb1_y >= height:
                    continue

                for nb1_x in range(nx - radius, nx + radius + 1):
                    if nb1_x < 0 or nb1_x >= width:
                        continue

                    if flags[nb1_x, nb1_y] == UNKNOWN:
                        continue

                    dir_y = ny - nb1_y
                    dir_x = nx - nb1_x

                    dir_length_square = dir_y ** 2 + dir_x ** 2
                    dir_length = sqrt(dir_length_square)

                    if dir_length > radius:
                        continue

                    direction = abs(dir_y * grad_y + dir_x * grad_x)
                    if direction == 0.0:
                        direction = MIN

                    factor = 1.0 / (1.0 + abs(dists[nb1_x, nb1_y] - dist))

                    distance = 1.0 / (dir_length * dir_length_square)

                    weight = abs(direction * distance * factor)

                    pixel_sum[:] += weight * img[nb1_x, nb1_y, :]

                    weight_sum += weight

            pixel_avg = pixel_sum / weight_sum

            src_img[nx, ny, :] = pixel_avg[:]

            flags[nx, ny] = BAND
            heapq.heappush(band, (dist, nx, ny))


drawing = False
ix, iy = -1, -1


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img1, (x, y), drawsize, (255, 255, 255), -1)
            cv2.circle(img2, (x, y), drawsize, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img1, (x, y), drawsize, (255, 255, 255), -1)
        cv2.circle(img2, (x, y), drawsize, (255, 255, 255), -1)


img = cv2.imread(in_path)
height, width = img.shape[0:2]
img2 = np.zeros((width, height, 3), np.uint8)

img1 = img.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img1)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# mask_img = cv2.imread(mask_path)
# cv2.imshow('mask', img2)

mask = img2[:, :, 0].astype(bool, copy=False)
inpaint(img, mask)
cv2.imshow('out', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
