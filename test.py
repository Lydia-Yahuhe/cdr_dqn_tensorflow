import numpy as np
import cv2


def cv2_demo_trackbar():
    def nothing(x):
        pass

    # 创建一个黑色的图像，一个窗口
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # 创建颜色变化的轨迹栏
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    # 为 ON/OFF 功能创建开关
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # 得到四条轨迹的当前位置
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]
    cv2.destroyAllWindows()


def cv2_tracks_bar():
    # 图像膨胀函数
    def img_dilated(img, d):
        # 定义 kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
        # 图像膨胀
        dilated = cv2.dilate(img, kernel)
        # 返回膨胀图片
        return dilated

    # 回调函数，因为只能传一个参数，不方便，所以pass
    def nothing(pos):
        pass

    # 读取图片
    img = cv2.imread("wuhan_base.jpg", 1)
    # 创建老窗口
    cv2.namedWindow('OldImg')
    # 绑定老窗口和滑动条（滑动条的数值）
    cv2.createTrackbar('D', 'OldImg', 1, 30, nothing)
    while True:
        # 提取滑动条的数值d
        d = cv2.getTrackbarPos('D', 'OldImg')
        # 滑动条数字传入函数img_dilated中，并且调用函数img_dilated
        dilated = img_dilated(img, d)
        # 绑定 img 和 dilated
        result = np.hstack([img, dilated])
        cv2.imshow('OldImg', result)
        # 设置推出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 关闭窗口
    cv2.destroyAllWindows()


cv2_tracks_bar()
