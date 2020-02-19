import cv2
import sys
import numpy as np


class Ellipse_fitting():
    def __init__(self, path, low_threshold, high_threshold):
        """Constructor.

               :param image, canny low thresh, canny high threshold
               """
        self.l_threshold = low_threshold
        self.h_threshold = high_threshold
        self.image = cv2.imread(path)
        self.f_contour = []

    def canny_on_channels(self):
        """ edge detection on three channels (B G R)
            detect contour on the detected edges

               :param
               """
        blue, green, red = cv2.split(self.image)
        # canny edge detection on b g and r channels
        canny_b = cv2.Canny(blue, self.l_threshold, self.h_threshold)
        canny_g = cv2.Canny(green, self.l_threshold, self.h_threshold)
        canny_r = cv2.Canny(red, self.l_threshold, self.h_threshold)

        edge = canny_b | canny_g | canny_r
        contour, _ = self.contour_on_edge(edge)
        return edge, contour

    def filter_contour(self, contour):
        con = [c for c in contour if len(c) > 50]
        self.f_contour = [c for c in con if cv2.arcLength(c, True) > 150]
        # self.f_contour = [c for c in self.f_contour if self.connected_contour(c)]

    def contour_on_edge(self, edge):
        return cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def visualization(self, image, name):
        """ visualization function
            display edge image , contours on original image
               """
        cv2.drawContours(self.image, self.f_contour, -1, (0, 255, 0), 3)
        cv2.imshow(name, image)
        cv2.imshow("contour", self.image)

    # cv2.imshow("new_canny", canny_new)

    def connected_contour(self, contours) -> bool:
        """
         checking if a contour is closed or not
        :param contours:
        :return: bool
        """

        first = contours[0][0]
        last = contours[len(contours) - 1][0]
        return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1

    def canny_after_filter(self, edges):
        edges_new = np.zeros((edges.shape[0], edges.shape[1]))
        for c in self.f_contour:
            x_sub = [a[0] for a in c]
            for i in range(0, len(x_sub)):
                edges_new[x_sub[i][1], x_sub[i][0]] = 255
            # print(x_sub[i][1], x_sub[i][0])
        return edges_new

    def grid_from_canny(self, canny):
        """
          make blocks of 100 *100 from edge image and do ellipse fitting on each blocks
        :param canny:
        :return:
        """
        imgheight, imgwidth = canny.shape
        padh = 100 - (imgheight % 100)
        padw = 100 - (imgwidth % 100)
        top, bottom = padh / 2, padh - (padh / 2)
        left, right = padw / 2, padw - (padw / 2)

        canny = cv2.copyMakeBorder(canny, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                   value=0)

        canny = np.pad(canny, (int(padh / 2), int(padw / 2)))
        for i in range(0, imgheight, 100):
            for j in range(0, imgwidth, 100):
                a = canny[i:i + 100, j:j + 100]
                self.ellispe_fitting(a)
            # self.visualization(a, "grid")
            # cv2.waitKey(0)

    def ellispe_fitting(self, grid):
        y, x = np.where(grid == 255)
        ellipse = cv2.fitEllipse(np.column_stack((x, y)))
        cv2.ellipse(grid, ellipse, (0, 255, 0), 2)
        self.visualization(grid, "ellipse")
        cv2.waitKey(0)


if __name__ == "__main__":
    image_path = sys.argv[1]
    w_d = Ellipse_fitting(image_path, 30, 60)
    canny, contour = w_d.canny_on_channels()
    w_d.filter_contour(contour)
    canny_new = w_d.canny_after_filter(canny)
    w_d.grid_from_canny(canny)
    w_d.visualization(canny, "canny")
    w_d.visualization(canny_new, "canny_new")
    cv2.waitKey(-1)
