import mss
import numpy as np
import cv2
from pathlib import Path
import random
import pydirectinput
import math
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
from win32api import GetAsyncKeyState

VK_CODE = {
    "A": 0x41, "B": 0x42, "C": 0x43, "D": 0x44, "E": 0x45, "F": 0x46, "G": 0x47, "H": 0x48, "I": 0x49, "J": 0x4A,
    "K": 0x4B, "L": 0x4C, "M": 0x4D, "N": 0x4E, "O": 0x4F, "P": 0x50, "Q": 0x51, "R": 0x52, "S": 0x53, "T": 0x54,
    "U": 0x55, "V": 0x56, "W": 0x57, "X": 0x58, "Y": 0x59, "Z": 0x5A, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39, "0": 0x30, "Enter": 0x0D, "Esc": 0x1B, "BackSpace": 0x08,
    "Tab": 0x09, " ": 0x20, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC, ";": 0xBA, "'": 0xDE, "`": 0xC0,
    ",": 0xBC, ".": 0xBE, "/": 0xBF, "CapsLock": 0x14, "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74,
    "F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B, "PrintScreen": 0x2C,
    "ScrollLock": 0x91, "Pause": 0x13, "Break": 0x13, "Insert": 0x2D, "Home": 0x24, "Pageup": 0x21, "Delete": 0x2E,
    "End": 0x23, "Pagedown": 0x22, "Right": 0x27, "Left": 0x25, "Down": 0x28, "Up": 0x26, "NumLock": 0x90,
    "keypad./": 0x6F, "keypad.*": 0x60, "keypad.-": 0x6D, "keypad.+": 0x6B, "keypad.enter": 0x6C, "keypad.1": 0x61,
    "keypad.2": 0x62, "keypad.3": 0x63, "keypad.4": 0x64, "keypad.5": 0x65, "keypad.6": 0x66, "keypad.7": 0x67,
    "keypad.8": 0x68, "keypad.9": 0x69, "keypad.0": 0x60, "keypad..": 0x6E, "Menu": 0x5D, "keypad.=": 0x92, "静音": 0xAD,
    "音量加": 0xAF, "音量减": 0xAE, "left_Ctrl": 0xA2, "left_Shift": 0xA0, "left_Alt": 0xA4, "left_Win": 0x5B,
    "right_Ctrl": 0xA3, "right_Shift": 0xA1, "right_Alt": 0xA5, "right_Win": 0x5C, "Ctrl": 0x11, "Shift": 0x10,
    "Alt": 0x12, "l_button": 1, "r_button": 2, "cancel": 3, "m_button": 4, }
VK_CODE.update({_k.lower(): _v for _k, _v in VK_CODE.items()})
VK_CODE.update({_k.upper(): _v for _k, _v in VK_CODE.items()})
key_l_button = VK_CODE["l_button"]
get_key_state = lambda key: GetAsyncKeyState(key) & 0x8000

SYS_MSG = [230, 173, 164, 232, 189, 175, 228, 187, 182, 228, 184, 186, 229, 133, 141, 232, 180, 185, 229, 188, 128,
           230, 186, 144, 232, 189, 175, 228, 187, 182, 44, 228, 187, 133, 228, 190, 155, 229, 173, 166, 228, 185,
           160, 228, 186, 164, 230, 181, 129, 228, 189, 191, 231, 148, 168, 239, 188, 140, 229, 166, 130, 230, 158,
           156, 228, 189, 160, 230, 152, 175, 232, 180, 173, 228, 185, 176, 229, 190, 151, 229, 136, 176, 231, 154,
           132, 239, 188, 140, 233, 130, 163, 228, 185, 136, 230, 129, 173, 229, 150, 156, 228, 189, 160, 232, 162,
           171, 233, 170, 151, 228, 186, 134, 10, 229, 188, 128, 230, 186, 144, 229, 156, 176, 229, 157, 128, 10,
           229, 155, 189, 229, 134, 133, 233, 149, 156, 229, 131, 143, 229, 156, 176, 229, 157, 128, 58, 32, 104,
           116, 116, 112, 115, 58, 47, 47, 104, 117, 98, 46, 102, 97, 115, 116, 103, 105, 116, 46, 111, 114, 103,
           47, 115, 111, 108, 111, 73, 105, 102, 101, 47, 65, 117, 116, 111, 83, 116, 114, 105, 107, 101, 10, 230,
           186, 144, 229, 156, 176, 229, 157, 128, 58, 32, 104, 116, 116, 112, 115, 58, 47, 47, 103, 105, 116, 104,
           117, 98, 46, 99, 111, 109, 47, 115, 111, 108, 111, 73, 105, 102, 101, 47, 65, 117, 116, 111, 83, 116,
           114, 105, 107, 101, 10]



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

def FOV_x(target_move, width):
    # actual_move = atan(target_move / base_len) * base_len  # 弧长
    h = width / 2 / math.tan(3.6103 / 2)
    actual_move = math.atan(target_move / h) * -83.54377746582 + -0.1544615477323532
    return actual_move

def FOV_y(target_move, width):
    # actual_move = atan(target_move / base_len) * base_len  # 弧长
    h = width / 2 / math.tan(3.5044732093811035 / 2)
    actual_move = math.atan(target_move / h) * -41.59797286987305 + -0.5091857314109802
    return actual_move


# dx = FOV_x(dx, 1366)
# dy = FOV_y(dy, 768)
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
        if get_key_state(key_l_button):  # 鼠标左键
            if len(boxs)==0:
                continue
            else:
                pydirectinput.moveTo(int((boxs[0,0]+boxs[0,2])/2),int((boxs[0,1]+boxs[0,3])/2),relative=False)
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