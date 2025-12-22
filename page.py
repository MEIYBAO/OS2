import tkinter as tk
from tkinter import ttk, messagebox
import random


# ---------------------------
# 核心：分页 + 位示图(0/1) + 页表
# ---------------------------
class Process:
    def __init__(self, pid, num_pages: int):
        self.pid = str(pid)
        self.num_pages = int(num_pages)
        self.page_table = {i: -1 for i in range(self.num_pages)}  # page -> frame or -1


class SimplePager:
    def __init__(self, frame_count=20):
        self.frame_count = int(frame_count)
        self.frames = [None] * self.frame_count      # None or (pid, page)
        self.bitmap = [0] * self.frame_count         # 0 free, 1 used
        self.processes = {}                          # pid -> Process

    def set_frames(self, n: int):
        self.frame_count = int(n)
        self.frames = [None] * self.frame_count
        self.bitmap = [0] * self.frame_count
        # 所有进程页表清空（重新装入意义）
        for p in self.processes.values():
            p.page_table = {i: -1 for i in range(p.num_pages)}

    def free_count(self) -> int:
        return sum(1 for b in self.bitmap if b == 0)

    def add_process(self, pid, num_pages: int):
        pid = str(pid)
        if pid in self.processes:
            raise ValueError("PID 已存在")
        p = Process(pid, num_pages)
        self.processes[pid] = p
        return p

    def allocate_pages_first_fit(self, pid) -> int:
        pid = str(pid)
        if pid not in self.processes:
            raise ValueError("未知 PID")
        p = self.processes[pid]
        allocated = 0

        for page in range(p.num_pages):
            if p.page_table[page] != -1:
                continue

            found = False
            for i in range(self.frame_count):
                if self.bitmap[i] == 0:
                    self.bitmap[i] = 1
                    self.frames[i] = (pid, page)
                    p.page_table[page] = i
                    allocated += 1
                    found = True
                    break

            if not found:
                p.page_table[page] = -1

        return allocated

    def deallocate_process(self, pid) -> int:
        pid = str(pid)
        if pid not in self.processes:
            return 0
        p = self.processes[pid]
        freed = 0

        for page, frame in p.page_table.items():
            if frame != -1 and 0 <= frame < self.frame_count:
                if self.frames[frame] and self.frames[frame][0] == pid and self.frames[frame][1] == page:
                    self.frames[frame] = None
                    self.bitmap[frame] = 0
                    freed += 1

        del self.processes[pid]
        return freed


# ---------------------------
# UI：截图同款布局（帧网格 + 进程列表 + 页表）
# ---------------------------
class PageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("基本分页存储管理")

        self.pager = SimplePager(frame_count=20)

        # 自动可视化参数
        self.auto_running = False
        self.auto_paused = False
        self.step_ms = 600
        self.pid_counter = 1
        self.max_procs = 10  # 自动运行时的进程上限，避免越堆越多

        # 网格显示参数（匹配截图：一行 10 个小方块更像）
        self.grid_cols = 10
        self.cell_w = 44
        self.cell_h = 30

        # 颜色：不同 PID 给不同底色（固定调色板，效果接近截图）
        self.palette = [
            "#ffb3b3", "#ffd699", "#b3d9ff", "#b3ffcc", "#e0ccff",
            "#ffe6b3", "#cceeff", "#f2b3ff", "#d9ffb3", "#b3f0ff"
        ]
        self.pid_color = {}  # pid -> color

        self._build_ui()
        self._refresh_all()

    # ---------- UI 构建 ----------
    def _build_ui(self):
        # 顶部控制条
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="内存页框数:").grid(row=0, column=0, sticky="w")
        self.frame_var = tk.IntVar(value=self.pager.frame_count)
        ttk.Spinbox(top, from_=1, to=256, width=6, textvariable=self.frame_var).grid(row=0, column=1, padx=(4, 8))
        ttk.Button(top, text="应用页框数", command=self.apply_frames).grid(row=0, column=2, padx=(0, 16))

        ttk.Label(top, text="进程 ID").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.pid_entry = ttk.Entry(top, width=10)
        self.pid_entry.grid(row=1, column=1, padx=(4, 8), pady=(6, 0), sticky="w")

        ttk.Label(top, text="页数").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.pages_entry = ttk.Entry(top, width=8)
        self.pages_entry.grid(row=1, column=3, padx=(4, 8), pady=(6, 0), sticky="w")

        ttk.Button(top, text="创建并分配", command=self.create_and_alloc).grid(row=1, column=4, padx=(0, 8), pady=(6, 0))
        ttk.Button(top, text="释放进程", command=self.deallocate_selected).grid(row=1, column=5, padx=(0, 16), pady=(6, 0))

        self.btn_auto_toggle = ttk.Button(top, text="停止自动可视化", command=self.toggle_auto)
        self.btn_auto_toggle.grid(row=1, column=6, padx=(0, 8), pady=(6, 0))

        self.btn_pause = ttk.Button(top, text="暂停", command=self.toggle_pause)
        self.btn_pause.grid(row=1, column=7, padx=(0, 8), pady=(6, 0))

        ttk.Label(top, text="步进(ms)").grid(row=0, column=3, sticky="e")
        self.step_var = tk.IntVar(value=self.step_ms)
        ttk.Spinbox(top, from_=100, to=3000, increment=100, width=8, textvariable=self.step_var, command=self._update_step)\
            .grid(row=0, column=4, sticky="w", padx=(4, 0))

        self.status_var = tk.StringVar(value="状态：自动运行中")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=6, columnspan=2, sticky="e")

        # 主体区域：左内存页框 + 右进程与页表
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # 左侧：内存页框（小方块网格）
        left = ttk.LabelFrame(body, text="内存页框")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        self.mem_canvas = tk.Canvas(left, bg="white", highlightthickness=0)
        self.mem_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # 右侧：进程与列表（两列：进程列表、页表）
        right = ttk.LabelFrame(body, text="进程与列表")
        right.pack(side="left", fill="both", expand=True, padx=0, pady=0)

        right_inner = ttk.Frame(right)
        right_inner.pack(fill="both", expand=True, padx=8, pady=8)

        # 进程列表
        proc_frame = ttk.Frame(right_inner)
        proc_frame.pack(side="left", fill="both", expand=False)

        ttk.Label(proc_frame, text="").pack()  # 占位，让布局更像截图
        self.proc_list = tk.Listbox(proc_frame, width=18, height=20)
        self.proc_list.pack(fill="both", expand=True)
        self.proc_list.bind("<<ListboxSelect>>", self.show_page_table)

        # 页表
        pt_frame = ttk.Frame(right_inner)
        pt_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.pt_tree = ttk.Treeview(pt_frame, columns=("page", "frame"), show="headings", height=20)
        self.pt_tree.heading("page", text="页号")
        self.pt_tree.heading("frame", text="页框号")
        self.pt_tree.column("page", width=120, anchor="center")
        self.pt_tree.column("frame", width=140, anchor="center")
        self.pt_tree.pack(fill="both", expand=True)

        # 默认启动自动运行（符合你“一直运行下去”的要求）
        self.auto_running = True
        self.auto_paused = False
        self.btn_auto_toggle.config(text="停止自动可视化")
        self.btn_pause.config(text="暂停")
        self._auto_loop()

    # ---------- 业务按钮 ----------
    def _update_step(self):
        try:
            self.step_ms = int(self.step_var.get())
        except Exception:
            self.step_ms = 600

    def apply_frames(self):
        if self.auto_running:
            messagebox.showwarning("提示", "自动可视化运行中，请先停止再修改页框数")
            return
        n = int(self.frame_var.get())
        self.pager.set_frames(n)
        self.pid_color.clear()
        self._refresh_all()

    def create_and_alloc(self):
        pid = self.pid_entry.get().strip()
        pages = self.pages_entry.get().strip()

        if not pid or not pages.isdigit():
            messagebox.showwarning("输入错误", "请填写 PID 和 页数(整数)")
            return

        pages = int(pages)
        if self.pager.free_count() < pages:
            messagebox.showwarning("内存不足", f"当前空闲页框 {self.pager.free_count()} 小于所需 {pages}，无法分配")
            return

        try:
            self.pager.add_process(pid, pages)
        except ValueError as e:
            messagebox.showwarning("错误", str(e))
            return

        self._ensure_pid_color(pid)
        self.pager.allocate_pages_first_fit(pid)

        self.pid_entry.delete(0, "end")
        self.pages_entry.delete(0, "end")
        self._refresh_all()

    def deallocate_selected(self):
        sel = self.proc_list.curselection()
        if not sel:
            messagebox.showwarning("提示", "请选择要释放的进程")
            return
        pid = self.proc_list.get(sel[0])
        self.pager.deallocate_process(pid)
        if pid in self.pid_color:
            del self.pid_color[pid]
        self._refresh_all()

    # ---------- 自动可视化 ----------
    def toggle_auto(self):
        self.auto_running = not self.auto_running
        if self.auto_running:
            self.status_var.set("状态：自动运行中")
            self.btn_auto_toggle.config(text="停止自动可视化")
            self._auto_loop()
        else:
            self.status_var.set("状态：手动/停止")
            self.btn_auto_toggle.config(text="开始自动可视化")

    def toggle_pause(self):
        if not self.auto_running:
            return
        self.auto_paused = not self.auto_paused
        self.btn_pause.config(text="继续" if self.auto_paused else "暂停")
        self.status_var.set("状态：自动运行中（暂停）" if self.auto_paused else "状态：自动运行中")

    def _auto_loop(self):
        if not self.auto_running:
            return

        if not self.auto_paused:
            self._auto_step()

        self._refresh_all()
        self.root.after(self.step_ms, self._auto_loop)

    def _auto_step(self):
        # 策略：一直运行下去
        free_frames = self.pager.free_count()
        pids = list(self.pager.processes.keys())

        # 没进程 -> 创建
        if not pids:
            self._auto_create_one()
            return

        # 内存紧张 or 进程太多 -> 释放
        low_threshold = max(2, self.pager.frame_count // 6)
        if free_frames <= low_threshold or len(pids) >= self.max_procs:
            pid = random.choice(pids)
            self.pager.deallocate_process(pid)
            self.pid_color.pop(pid, None)
            self.status_var.set(f"状态：自动运行中 | 释放进程 {pid}")
            return

        # 否则：创建为主，释放为辅
        if random.random() < 0.72:
            self._auto_create_one()
        else:
            pid = random.choice(pids)
            self.pager.deallocate_process(pid)
            self.pid_color.pop(pid, None)
            self.status_var.set(f"状态：自动运行中 | 释放进程 {pid}")

    def _auto_create_one(self):
        free_frames = self.pager.free_count()
        if free_frames <= 0:
            return

        pid = str(self.pid_counter)
        self.pid_counter += 1

        # 页数：1 ~ min(空闲, 总帧/3)，更像实验演示
        max_need = min(free_frames, max(1, self.pager.frame_count // 3))
        pages = random.randint(1, max_need)

        try:
            self.pager.add_process(pid, pages)
            self._ensure_pid_color(pid)
            self.pager.allocate_pages_first_fit(pid)
            self.status_var.set(f"状态：自动运行中 | 创建进程 {pid}（{pages} 页）并分配")
        except ValueError:
            pass

    # ---------- 列表与页表 ----------
    def show_page_table(self, evt=None):
        for iid in self.pt_tree.get_children():
            self.pt_tree.delete(iid)

        sel = self.proc_list.curselection()
        if not sel:
            return

        pid = self.proc_list.get(sel[0])
        p = self.pager.processes.get(pid)
        if not p:
            return

        for page, frame in p.page_table.items():
            self.pt_tree.insert("", "end", values=(page, frame))

    # ---------- 内存页框网格绘制（截图同款） ----------
    def _ensure_pid_color(self, pid: str):
        if pid in self.pid_color:
            return
        # 用 pid 的哈希/计数给一个稳定颜色
        idx = (len(self.pid_color)) % len(self.palette)
        self.pid_color[pid] = self.palette[idx]

    def _draw_memory_grid(self):
        self.mem_canvas.delete("all")

        n = self.pager.frame_count
        cols = self.grid_cols
        rows = (n + cols - 1) // cols

        # 计算画布需要的最小尺寸
        pad = 8
        gap = 6
        w = pad * 2 + cols * self.cell_w + (cols - 1) * gap
        h = pad * 2 + rows * self.cell_h + (rows - 1) * gap

        self.mem_canvas.config(scrollregion=(0, 0, w, h))

        for i in range(n):
            r = i // cols
            c = i % cols
            x0 = pad + c * (self.cell_w + gap)
            y0 = pad + r * (self.cell_h + gap)
            x1 = x0 + self.cell_w
            y1 = y0 + self.cell_h

            owner = self.pager.frames[i]
            if owner is None:
                fill = "#ffffff"
                text = str(i)
                outline = "#888888"
            else:
                pid, page = owner
                fill = self.pid_color.get(pid, "#d9edf7")
                text = f"{pid}:{page}"
                outline = "#666666"

            # 方块
            self.mem_canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=1)
            # 文本（居中）
            self.mem_canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=text, font=("Consolas", 10))

    def _refresh_all(self):
        # 刷新内存格子
        self._draw_memory_grid()

        # 刷新进程列表
        cur = self.proc_list.curselection()
        self.proc_list.delete(0, "end")
        for pid in sorted(self.pager.processes.keys(), key=lambda x: int(x) if x.isdigit() else x):
            self.proc_list.insert("end", pid)

        # 尝试保持选中
        if self.proc_list.size() > 0:
            if cur:
                idx = min(cur[0], self.proc_list.size() - 1)
                self.proc_list.select_set(idx)
            else:
                self.proc_list.select_set(0)

        self.show_page_table()


if __name__ == "__main__":
    root = tk.Tk()
    app = PageApp(root)
    root.geometry("900x520")
    root.mainloop()
