import cv2

ch = 1

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mst = cv2.imread('moustache.png')
hat = cv2.imread('hat.png')
dog = cv2.imread('dog.png')


def apply_mst(filter, frame, x, y, w, h):
    mst_w = int(w * 0.50)
    mst_h = int(h * 0.15)
    off_x = int(w * 0.25)
    off_y = int(h * 0.65)
    filter = cv2.resize(filter, (mst_w, mst_h))
    for i in range(mst_h):
        for j in range(mst_w):
            for k in range(3):
                if filter[i][j][k] < 235:  # to remove background white
                    frame[y + i + off_y][x + j + off_x][k] = filter[i][j][k]
    return frame



def apply_hat(filter, frame, x, y, w, h):
    mst_w = int(w * 1.5)
    mst_h = int(h * 0.9)
    off_x = int(w * -0.25)
    off_y = int(h * -0.65)
    filter = cv2.resize(filter, (mst_w, mst_h))
    for i in range(mst_h):
        for j in range(mst_w):
            for k in range(3):
                if filter[i][j][k] < 235:  # to remove background white
                    frame[y + i + off_y][x + j + off_x][k] = filter[i][j][k]
    return frame

def apply_dog(filter, frame, x, y, w, h):
    mst_w = int(w * 1.5)
    mst_h = int(h * 2.5)
    off_x = int(w * -0.35)
    off_y = int(h * -0.65)
    filter = cv2.resize(filter, (mst_w, mst_h))
    for i in range(mst_h):
        for j in range(mst_w):
            for k in range(3):
                if filter[i][j][k] < 235:  # to remove background white
                    frame[y + i + off_y][x + j + off_x][k] = filter[i][j][k]
    return frame


video_capture = cv2.VideoCapture(0)
while True:
    if not video_capture.isOpened():
        exit()
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    print(len(faces))
    print(faces)
    for (x, y, w, h) in faces:
        if ch == 0:
            frame = apply_mst(mst, frame, x, y, w, h)
        elif ch == 1:
            frame = apply_hat(hat, frame, x, y, w, h)
        else:
            frame = apply_dog(dog, frame, x, y, w, h)

    cv2.imshow('Filter', frame)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
