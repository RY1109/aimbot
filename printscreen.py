import mss
import numpy as np
import cv2
from pathlib import Path
import random
import pydirectinput
from win32api import GetAsyncKeyState
import time
import matplotlib.pyplot as plt
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize('person', 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, 'person', (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot(frame, labels, boxes):
    for label, box in zip(labels, boxes):
        # label = '%s %.2f'
        plot_one_box(box, frame, label=label, color=(0, 255, 0), line_thickness=3)

def grab_screen(
        weights=ROOT / 'runs/train/exp26/weights/best.pt',  # model.pt path(s)
        device=torch.device('cuda'),
        conf_thres = 0.6,
        iou_thres = 0.5,
        classes=(0,),
        agnostic_nms = False,
        max_det = 50
):
    half = False
    model = DetectMultiBackend(weights, device=device, dnn=True)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = (1080,1920)
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    img_size = check_img_size(imgsz, s=stride)  # check image size

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
        img0 = sct.grab(monitor=monitor)
        img0 = np.array(img0).astype('float32')
        img = img0[:,:,0:3]
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = letterbox(img, new_shape=img_size, stride=stride, auto=True)[0].swapaxes(0,2).swapaxes(1,2)
        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred= model(im, augment=False, visualize=False)
        det = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                  max_det=max_det)[0]
        boxs = det[:,:4]
        labels = det[:,4:5]
        if pydirectinput.MOUSEEVENTF_LEFTCLICK:
            pydirectinput.moveTo(int((boxs[0,0]+boxs[0,2])/2),int((boxs[0,1]+boxs[0,3])/2))
        print(det)
        # if (flag%200000)==1:
        #     folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        #     plt.imsave(folder_name,img)
        #     flag = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 根据窗口大小设置我们的图片大小
        cv2.resizeWindow(window_name, RESIZE_WIN_WIDTH, RESIZE_WIN_HEIGHT)
        plot(img0, labels, boxs)
        cv2.imshow(window_name, img0)

        k = cv2.waitKey(1)
        flag +=1
        if k % 256 == 27:  # ESC
            cv2.destroyAllWindows()
            exit('ESC ...')

if __name__ == '__main__':
    img = grab_screen()
    pass