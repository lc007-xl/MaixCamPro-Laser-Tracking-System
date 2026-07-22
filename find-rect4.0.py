from maix import camera, display, image, app
from maix.peripheral import uart
import cv2
import numpy as np


IMG_W, IMG_H = 320, 240
PROC_W, PROC_H = 160, 120          
SCALE_X = IMG_W / PROC_W
SCALE_Y = IMG_H / PROC_H

cam = camera.Camera(IMG_W, IMG_H, fps=60)
cam.skip_frames(10)
disp = display.Display()
serial = uart.UART("/dev/ttyS0", 115200)


CONFIRM_FRAMES = 3     
LOST_BUFFER    = 6     
DEAD_ZONE      = 2     
MIN_AREA       = 100   
MAX_AREA_RATIO = 0.6
ACQ_JUMP       = 60    

KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


_np_path = None
def to_numpy(img):
    global _np_path
    if _np_path is None:
        try:
            a = np.array(img, copy=False)
            if a.shape == (IMG_H, IMG_W, 3):
                _np_path = 1
                return a
        except Exception:
            pass
        _np_path = 2
    if _np_path == 1:
        return np.array(img, copy=False)
    raw = img.tobytes()
    return np.frombuffer(raw, dtype=np.uint8).reshape(IMG_H, IMG_W, 3)


class Tracker:

    def __init__(self):
        self.ready = False
        self.x = self.y = self.w = self.h = 0.0
        self.vx = self.vy = 0.0

    def correct(self, x, y, w, h):
       
        if not self.ready:
            self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)
            self.vx = self.vy = 0.0
            self.ready = True
            return
        ocx, ocy = self.x + self.w / 2, self.y + self.h / 2
        dx = (x + w / 2) - ocx
        dy = (y + h / 2) - ocy
        dist = (dx * dx + dy * dy) ** 0.5
        a = min(0.55, 0.15 + 0.02 * dist)   # 静止0.15 → 25px位移0.55
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        self.w += a * (w - self.w)
        self.h += a * (h - self.h)
        # 速度EMA, 供丢失缓冲帧外推
        self.vx = 0.6 * self.vx + 0.4 * (self.x + self.w / 2 - ocx)
        self.vy = 0.6 * self.vy + 0.4 * (self.y + self.h / 2 - ocy)

    def predict(self):
       
        self.x = min(max(self.x + self.vx, 0.0), IMG_W - self.w)
        self.y = min(max(self.y + self.vy, 0.0), IMG_H - self.h)
        self.vx *= 0.8
        self.vy *= 0.8

    def rect(self):
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def reset(self):
        self.ready = False
        self.vx = self.vy = 0.0


def pick_contour(binary, relaxed):
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill_th = 0.40 if relaxed else 0.55
    lo, hi  = (0.30, 3.3) if relaxed else (0.40, 2.5)
    max_area = binary.shape[0] * binary.shape[1] * MAX_AREA_RATIO
    best_score, best_rect = 0.0, None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA or area > max_area or h == 0:
            continue
        ratio = w / h
        if ratio < lo or ratio > hi:
            continue
        fill = cv2.contourArea(cnt) / area
        peri = cv2.arcLength(cnt, True)
        is_quad = len(cv2.approxPolyDP(cnt, 0.03 * peri, True)) == 4
        if fill < fill_th and not is_quad:
            continue
        score = area * fill * (1.5 if is_quad else 1.0)
        score *= 1.0 / (1.0 + abs(ratio - 1.0))   
        if score > best_score:
            best_score, best_rect = score, (x, y, w, h)
    return best_rect


def detect(np_img, roi=None, relaxed=False):

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (PROC_W, PROC_H), interpolation=cv2.INTER_NEAREST)
    ox = oy = 0
    if roi is not None:
        x0, y0, x1, y1 = roi
        gray = gray[y0:y1, x0:x1]
        ox, oy = x0, y0
    if gray.std() < 28:                 
        gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, KERNEL5,
                              iterations=2 if relaxed else 1)  # 锁定后加强闭运算修复残缺色块
    rect = pick_contour(binary, relaxed)
    if rect is None and relaxed:
        # 备用阈值: OTSU在强反光/欠曝下失效时, 局部自适应阈值再试一次
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, KERNEL5)
        rect = pick_contour(binary, relaxed)
    if rect is None:
        return None
    x, y, w, h = rect
    return (int((x + ox) * SCALE_X), int((y + oy) * SCALE_Y),
            int(w * SCALE_X), int(h * SCALE_Y))


# ===== 状态机: LOST -> ACQ(确认计数) -> LOCK(带丢失缓冲) =====
tracker = Tracker()
locked = False
confirm_count = 0
lost_count = 0
last_out = None

while not app.need_exit():
    img = cam.read()
    if img is None:
        continue
    np_img = to_numpy(img)

    # ---- 1. 检测: 每帧都检测; 锁定后只搜预测位置附近小窗口(更快+天然抗干扰) ----
    roi = None
    if locked and lost_count <= 2:      # 刚丢1~2帧仍在小窗找; 丢更久回退全图搜索
        pcx = (tracker.x + tracker.w / 2 + tracker.vx) / SCALE_X
        pcy = (tracker.y + tracker.h / 2 + tracker.vy) / SCALE_Y
        r = max(30, int((tracker.w + tracker.h) / 2 / SCALE_X))          # 随目标尺寸
        r += int((abs(tracker.vx) + abs(tracker.vy)) / SCALE_X) + lost_count * 10  # 随速度/丢失时长
        x0, y0 = max(0, int(pcx - r)), max(0, int(pcy - r))
        x1, y1 = min(PROC_W, int(pcx + r)), min(PROC_H, int(pcy + r))
        if x1 - x0 >= 20 and y1 - y0 >= 20:
            roi = (x0, y0, x1, y1)

    target = detect(np_img, roi=roi, relaxed=locked)

    # ---- 2. 状态机更新 ----
    if locked:
        if target is not None:
            # 动态门限: 目标越大/速度越快/丢得越久, 允许的跳变越大
            gate = max(80.0, (tracker.w + tracker.h) / 2) \
                   + 2 * (abs(tracker.vx) + abs(tracker.vy)) + lost_count * 15
            pcx = tracker.x + tracker.w / 2 + tracker.vx
            pcy = tracker.y + tracker.h / 2 + tracker.vy
            tcx = target[0] + target[2] / 2
            tcy = target[1] + target[3] / 2
            if abs(tcx - pcx) > gate or abs(tcy - pcy) > gate:
                target = None           # 门限外视为别的方块, 不打断当前跟踪
        if target is not None:
            lost_count = 0
            tracker.correct(*target)
        else:
            lost_count += 1
            if lost_count <= LOST_BUFFER:
                tracker.predict()       # 缓冲期: 匀速外推顶替观测, 保持LOCK连续输出
            else:
                locked = False          # 缓冲耗尽才判定真正丢失
                confirm_count = 0
                tracker.reset()
    else:
        if target is not None:
            if tracker.ready:
                dx = abs(target[0] + target[2] / 2 - (tracker.x + tracker.w / 2))
                dy = abs(target[1] + target[3] / 2 - (tracker.y + tracker.h / 2))
                if dx > ACQ_JUMP or dy > ACQ_JUMP:   # 确认期跳变过大 -> 新目标重新计数
                    tracker.reset()
                    confirm_count = 0
            tracker.correct(*target)     # 确认期也平滑, 锁定瞬间坐标不跳变
            confirm_count += 1
            if confirm_count >= CONFIRM_FRAMES:
                locked = True
                lost_count = 0
                last_out = None
        else:
            confirm_count = 0
            tracker.reset()

    # ---- 3. 输出 (锁定期间每帧都发平滑值, 下位机数据不断流) ----
    if locked:
        x, y, w, h = tracker.rect()
        # 输出死区: 四个值变化都≤DEAD_ZONE则沿用旧值, 串口/画面不再高频微抖
        if last_out is not None and \
           all(abs(a - b) <= DEAD_ZONE for a, b in zip((x, y, w, h), last_out)):
            x, y, w, h = last_out
        last_out = (x, y, w, h)
        cx, cy = x + w // 2, y + h // 2
        col = image.COLOR_GREEN if lost_count == 0 else image.COLOR_ORANGE  # 橙色=预测中
        img.draw_rect(x, y, w, h, col, thickness=2)
        img.draw_cross(cx, cy, image.COLOR_BLUE, size=10, thickness=2)
        serial.write_str("%d,%d,%d,%d\n" % (x, y, w, h))
    elif target is not None:            # 确认期黄框
        x, y, w, h = tracker.rect() if tracker.ready else target
        img.draw_rect(x, y, w, h, image.COLOR_YELLOW, thickness=2)

    # ---- 4. HUD ----
    mx, my = IMG_W // 2, IMG_H // 2
    img.draw_line(0, my, IMG_W, my, image.COLOR_GRAY)
    img.draw_line(mx, 0, mx, IMG_H, image.COLOR_GRAY)
    img.draw_rect(mx - 15, my - 15, 30, 30, image.COLOR_YELLOW)
    if locked:
        s, c = ("LOCK" if lost_count == 0 else "HOLD"), image.COLOR_GREEN
    elif confirm_count > 0:
        s, c = "ACQ", image.COLOR_YELLOW
    else:
        s, c = "LOST", image.COLOR_RED
    img.draw_string(0, 0, s, color=c, scale=1.5)

    disp.show(img)
