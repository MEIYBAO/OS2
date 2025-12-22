import time
import threading
import tkinter as tk
from collections import deque

# ---------------------------
# 参数（对齐你的 pv.cpp 风格）
# ---------------------------
BufferSize = 5
p_num = 2
con_num = 5
STEP_MS = 600  # 每一步动作可视化节奏（毫秒）


# ---------------------------
# 可计数信号量（用于 GUI 显示 empty/full）
# ---------------------------
class CountingSemaphore:
    """
    用 threading.Condition 实现的可计数信号量：
    - acquire() 阻塞直到 value > 0
    - release() value += 1 并唤醒
    """
    def __init__(self, initial: int):
        self._value = initial
        self._cond = threading.Condition()

    def acquire(self):
        with self._cond:
            while self._value <= 0:
                self._cond.wait()
            self._value -= 1

    def release(self):
        with self._cond:
            self._value += 1
            self._cond.notify()

    def value(self) -> int:
        with self._cond:
            return self._value


# ---------------------------
# Tkinter 可视化应用
# ---------------------------
class PVApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Producer-Consumer (PV) - Tkinter Visualization")

        # 共享数据（缓冲区 + 指针）
        self.buffer = [0] * BufferSize
        self.in_idx = 0
        self.out_idx = 0

        # PV：empty / full(product) / mutex
        self.empty = CountingSemaphore(BufferSize)
        self.full  = CountingSemaphore(0)
        self.mutex = CountingSemaphore(1)

        # 日志（线程写入，UI线程读取）
        self.log_buf = deque(maxlen=10)
        self.log_lock = threading.Lock()

        # --- UI 布局 ---
        self.canvas = tk.Canvas(root, width=700, height=220, bg="white")
        self.canvas.pack(padx=10, pady=(10, 6))

        info_frame = tk.Frame(root)
        info_frame.pack(fill="x", padx=10)

        self.lbl_sem = tk.Label(info_frame, text="", anchor="w", font=("Consolas", 11))
        self.lbl_sem.pack(fill="x")

        self.txt_log = tk.Text(root, height=8, width=80, font=("Consolas", 10))
        self.txt_log.pack(padx=10, pady=(6, 10), fill="both", expand=True)
        self.txt_log.configure(state="disabled")

        # 绘制参数
        self.cell_w = 90
        self.cell_h = 70
        self.start_x = 40
        self.start_y = 80

        # 启动线程
        self._start_threads()

        # UI 定时刷新
        self._schedule_refresh()

        # 关闭窗口时停止
        self._stop_evt = threading.Event()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------
    # 线程逻辑（生产者/消费者）
    # ---------------------------
    def _start_threads(self):
        for i in range(1, p_num + 1):
            t = threading.Thread(target=self._producer, args=(i,), daemon=True)
            t.start()
        for i in range(1, con_num + 1):
            t = threading.Thread(target=self._consumer, args=(i,), daemon=True)
            t.start()

    def _sleep_step(self):
        time.sleep(STEP_MS / 1000.0)

    def _log(self, msg: str):
        with self.log_lock:
            self.log_buf.appendleft(msg)

    def _producer(self, pid: int):
        x = 1
        while True:
            # PV
            self.empty.acquire()
            self.mutex.acquire()

            # 临界区：写入 1（资源存在），也可写入序号 x 方便展示
            self.buffer[self.in_idx] = 1
            self._log(f"[P{pid}] put -> slot {self.in_idx}")
            self.in_idx = (self.in_idx + 1) % BufferSize

            self._sleep_step()

            self.mutex.release()
            self.full.release()

            x += 1

    def _consumer(self, cid: int):
        while True:
            # PV
            self.full.acquire()
            self.mutex.acquire()

            # 临界区：取出并清空
            self.buffer[self.out_idx] = 0
            self._log(f"[C{cid}] get <- slot {self.out_idx}")
            self.out_idx = (self.out_idx + 1) % BufferSize

            self._sleep_step()

            self.mutex.release()
            self.empty.release()

    # ---------------------------
    # UI 刷新（主线程）
    # ---------------------------
    def _schedule_refresh(self):
        self._refresh_ui()
        self.root.after(80, self._schedule_refresh)  # 约 12.5 FPS

    def _refresh_ui(self):
        # 1) 更新画布（填涂颜色表示资源）
        self.canvas.delete("all")

        # 标题
        self.canvas.create_text(350, 20, text="Producer-Consumer Buffer ",
                                font=("Arial", 14))

        # in/out 指针标签
        for i in range(BufferSize):
            x1 = self.start_x + i * self.cell_w
            y1 = self.start_y
            x2 = x1 + self.cell_w - 10
            y2 = y1 + self.cell_h

            occupied = (self.buffer[i] != 0)

            # 填涂颜色：有资源=绿色填充；空=浅灰
            fill = "#5CB85C" if occupied else "#F5F5F5"
            outline = "#333333"

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2)

            # 槽位编号
            self.canvas.create_text((x1 + x2) / 2, y2 + 18, text=f"slot {i}", font=("Consolas", 10))

            # in/out 指针
            if i == self.in_idx:
                self.canvas.create_text((x1 + x2) / 2, y1 - 18, text="in", font=("Consolas", 12, "bold"))
            if i == self.out_idx:
                self.canvas.create_text((x1 + x2) / 2, y2 + 38, text="out", font=("Consolas", 12, "bold"))


        # 3) 更新日志
        with self.log_lock:
            logs = list(self.log_buf)

        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.insert("1.0", "\n".join(logs))
        self.txt_log.configure(state="disabled")

    def on_close(self):
        # daemon 线程会随主进程退出；这里做个标记（如你后面想优雅退出可扩展）
        self._stop_evt.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PVApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
