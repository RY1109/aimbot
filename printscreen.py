import mss
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def grab_screen():
    flag = 0
    sct = mss.mss()
    screen_width = 1920  # 屏幕的宽
    screen_height = 1080  # 屏幕的高
    GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT = 0,0,1920,1080 # 游戏内截图区域
    RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT = screen_width // 3, screen_height // 3  # 显示窗口大小
    monitor = {
        'left': GAME_LEFT,
        'top': GAME_TOP,
        'width': GAME_WIDTH,
        'height': GAME_HEIGHT
    }
    window_name = 'test'
    while True:
        img = sct.grab(monitor=monitor)
        img = np.array(img)
        if (flag%200000)==1:
            folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            plt.imsave(folder_name,img)
            flag = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 根据窗口大小设置我们的图片大小
        cv2.resizeWindow(window_name, RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1)
        flag +=1
        if k % 256 == 27:  # ESC
            cv2.destroyAllWindows()
            exit('ESC ...')

if __name__ == '__main__':
    img = grab_screen()
    pass