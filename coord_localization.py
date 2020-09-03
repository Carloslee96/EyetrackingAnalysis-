import cv2
# import pandas as pd

def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 255, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness = 1)
        cv2.imshow("image", img)

img = cv2.imread("D:\Pycharm\Project\plot_all_data\input_img\person3_000001.jpg")
cv2.namedWindow("image")
cv2.imshow("image", img)
cv2.resizeWindow("image", 1280, 720)
cv2.setMouseCallback("image", mouse)

cv2.waitKey(0)
cv2.destroyAllWindows()

