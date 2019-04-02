import cv2
import numpy as np


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    clicked_points = param["clicked_points"]
    src = param["src"]
    if event == cv2.EVENT_LBUTTONDOWN:
        print("click at point: x=%d, y=%d" % (x, y))
        cv2.circle(src, (x, y), 3, (0, 255, 0), thickness=-1)
        clicked_points.append([x, y])


def getClickedPointCoord(src):
    h, w = src.shape[:2]
    clicked_points = []
    # callback function, get the coordinates where mouse click
    cv2.namedWindow('src')
    cv2.setMouseCallback("src", on_EVENT_LBUTTONDOWN, param={"src": src, "clicked_points": clicked_points})

    while (True):
        try:
            cv2.imshow("src", src)
            if cv2.waitKey(100) & 0xFF == 27:  # esc exit
                break
        except Exception:
            cv2.destroyAllWindows()
            break
    print(clicked_points)
    return clicked_points


if __name__ == "__main__":
    test_pic_path = r"./picture/1_1.jpg"
    src = cv2.imread(test_pic_path)
    h = int(src.shape[0] / 2)
    w = int(src.shape[1] / 2)
    print("read source image and resize to (width=%d, height=%d)" % (w, h))

    src = cv2.resize(src, (w, h))
    clicked_points = getClickedPointCoord(src)

    pts1 = np.float32(clicked_points)
    pts2 = np.float32([[0, 0], [w/2, 0], [w, 0], [0, h], [w/2, h], [w, h]])
    M, _ = cv2.findHomography(pts1, pts2)
    print(M)

    dst = cv2.warpPerspective(src, M, (w, h))

    cv2.imshow("dst", dst)
    cv2.waitKey(0)