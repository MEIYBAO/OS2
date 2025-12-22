import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
from collections import deque, namedtuple
import matplotlib
matplotlib.use('TkAgg')
# 设置中文字体优先（Windows 常见），并确保负号正常显示
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# 进程数据结构
class Process:
    def __init__(self, pid, arrive, service):
        self.pid = str(pid)
        self.arrive = int(arrive)
        self.service = int(service)
        self.remain = int(service)
        self.start_time = None
        self.finish_time = None
        # 剩余时间片（用于剥夺后恢复时继续当前时间片）
        self.slice_remain = None

# 调度器和 GUI 应用
class MFQApp:
    def __init__(self, root):
        self.root = root
        self.root.title('MFQ 多级反馈队列')

        # 默认时间片
        self.timeslice = [2, 4, 8]
        self.queue_num = 3
        self.preemptible = tk.BooleanVar(value=True)

        # 进程列表（输入），调度数据
        self.proc_list = []  # 未到达的进程
        self.time = 0
        self.running = False
        self.pause = False

        # 就绪队列
        self.ready = [deque() for _ in range(self.queue_num)]
        # 当前运行进程
        self.current = None
        self.current_level = None
        self.current_slice_remain = 0

        # 甘特数据： list of (pid, start, duration, level)
        self.gantt = []
        # 已完成的进程列表（用于右侧显示）
        self.finished = []

        # 颜色映射：为每个进程分配唯一颜色
        self.color_map = {}
        # 使用 tab20 这样颜色较多且辨识度好
        self.cmap = plt.get_cmap('tab20')

        self._build_ui()
        self._build_plot()

        # lock
        self.lock = threading.Lock()

    def _build_ui(self):
        frm = ttk.Frame(self.root)
        frm.pack(side='top', fill='x', padx=6, pady=6)
        self.top_frame = frm

        # 可调整队列数量（修改时要求仿真停止）
        ttk.Label(frm, text='队列数:').grid(row=0, column=0, sticky='w')
        self.queue_num_var = tk.IntVar(value=self.queue_num)
        self.spin_queue = tk.Spinbox(frm, from_=1, to=8, textvariable=self.queue_num_var, width=4)
        self.spin_queue.grid(row=0, column=1, sticky='w')
        ttk.Button(frm, text='应用队列数', command=self.apply_queue_change).grid(row=0, column=2, padx=6)

        # 时间片输入，保存标签引用以便在更改队列数时更新显示
        self.ts_label = ttk.Label(frm, text=f"时间片 (队列{','.join(str(i) for i in range(self.queue_num))}):")
        self.ts_label.grid(row=1, column=0)
        self.ts_vars = [tk.IntVar(value=self.timeslice[i]) for i in range(self.queue_num)]
        self.ts_entries = []
        for i in range(self.queue_num):
            e = ttk.Entry(frm, textvariable=self.ts_vars[i], width=4)
            e.grid(row=1, column=1+i)
            self.ts_entries.append(e)

        ttk.Checkbutton(frm, text='可剥夺 (preemptible)', variable=self.preemptible).grid(row=0, column=4, padx=8)

        # 添加进程和导入示例
        ttk.Label(frm, text='进程ID').grid(row=2, column=0)
        self.entry_id = tk.Entry(frm, width=6)
        self.entry_id.grid(row=2, column=1)
        ttk.Label(frm, text='到达时间').grid(row=2, column=2)
        self.entry_arr = tk.Entry(frm, width=6)
        self.entry_arr.grid(row=2, column=3)
        ttk.Label(frm, text='服务时间').grid(row=2, column=4)
        self.entry_srv = tk.Entry(frm, width=6)
        self.entry_srv.grid(row=2, column=5)
        ttk.Button(frm, text='添加进程', command=self.add_job).grid(row=2, column=6, padx=6)
        ttk.Button(frm, text='导入示例', command=self.import_example).grid(row=2, column=7, padx=6)

        # 控制按钮
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=3, column=0, columnspan=8, pady=6)
        ttk.Button(btn_frame, text='开始', command=self.start).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text='暂停/继续', command=self.toggle_pause).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text='重置', command=self.reset).grid(row=0, column=2, padx=4)
        ttk.Button(btn_frame, text='导出甘特图', command=self.export_gantt).grid(row=0, column=3, padx=4)

        # 列表显示队列
        self.queues_frame = ttk.Frame(self.root)
        self.queues_frame.pack(side='left', fill='y', padx=6, pady=6)
        self.queue_listboxes = []
        for i in range(self.queue_num):
            lbl = ttk.Label(self.queues_frame, text=f'就绪队列 {i} (TS={self.ts_vars[i].get()})')
            lbl.pack(anchor='w')
            lb = tk.Listbox(self.queues_frame, width=30, height=6)
            lb.pack(pady=2)
            self.queue_listboxes.append(lb)

        # 当前时间和状态
        status_frame = ttk.Frame(self.queues_frame)
        status_frame.pack(fill='x', pady=6)
        self.time_var = tk.StringVar(value='时间: 0')
        ttk.Label(status_frame, textvariable=self.time_var).pack(anchor='w')
        self.cur_var = tk.StringVar(value='当前: 空闲')
        ttk.Label(status_frame, textvariable=self.cur_var).pack(anchor='w')

    def apply_queue_change(self):
        # 只能在仿真停止时修改队列数
        if self.running:
            messagebox.showwarning('禁止修改', '请先停止仿真后再修改队列数')
            return
        try:
            n = int(self.queue_num_var.get())
        except Exception:
            messagebox.showwarning('输入错误', '队列数必须为整数')
            return
        if n < 1:
            messagebox.showwarning('输入错误', '队列数至少为1')
            return
        if n == self.queue_num:
            return

        old = self.queue_num
        old_ready = self.ready

        # build new ready list
        new_ready = [deque() for _ in range(n)]
        # move existing queued processes into new_ready
        for i in range(min(old, n)):
            # keep order
            while old_ready[i]:
                new_ready[i].append(old_ready[i].popleft())
        if n > old:
            # nothing more to move
            pass
        else:
            # n < old: move items from queues >= n into last queue (n-1)
            for i in range(n, old):
                while old_ready[i]:
                    new_ready[n-1].append(old_ready[i].popleft())

        # if current exists and its level >= n, move it logically to last level
        if self.current and self.current_level is not None and self.current_level >= n:
            self.current_level = n-1

        self.ready = new_ready

        # update timeslice variables and listboxes
        old_ts = list(self.timeslice)
        self.timeslice = [old_ts[i] if i < len(old_ts) else 2 for i in range(n)]
        self.queue_num = n
        # rebuild ts_vars
        self.ts_vars = [tk.IntVar(value=self.timeslice[i]) for i in range(self.queue_num)]
        # rebuild top-frame timeslice entry widgets
        try:
            for e in getattr(self, 'ts_entries', []):
                e.destroy()
        except Exception:
            pass
        self.ts_entries = []
        for i in range(self.queue_num):
            e = ttk.Entry(self.top_frame, textvariable=self.ts_vars[i], width=4)
            e.grid(row=1, column=1+i)
            self.ts_entries.append(e)

        # update the timeslice label to reflect new queue indices
        try:
            self.ts_label.config(text=f"时间片 (队列{','.join(str(i) for i in range(self.queue_num))}):")
        except Exception:
            pass

        # rebuild queue listboxes UI
        for child in self.queues_frame.winfo_children():
            child.destroy()
        self.queue_listboxes = []
        for i in range(self.queue_num):
            lbl = ttk.Label(self.queues_frame, text=f'就绪队列 {i} (TS={self.ts_vars[i].get()})')
            lbl.pack(anchor='w')
            lb = tk.Listbox(self.queues_frame, width=30, height=6)
            lb.pack(pady=2)
            self.queue_listboxes.append(lb)

        # update plot limits
        self.ax.set_ylim(0, self.queue_num*10)
        self._refresh_ui()

    def _build_plot(self):
        # matplotlib 甘特图，并在右侧显示已完成队列
        # 使用一个 container frame 承载画布和完成列表
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side='right', fill='both', expand=1)

        self.fig = Figure(figsize=(8,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(0, self.queue_num*10)
        self.ax.set_xlabel('时间')
        self.ax.set_yticks([])
        self.ax.set_title('甘特图')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side='left', fill='both', expand=1)

        # 右侧的已完成进程显示区（表格：PID, 到达, 服务, 完成, 周转, 带权周转）
        self.finished_frame = ttk.Frame(self.plot_frame)
        self.finished_frame.pack(side='right', fill='y', padx=6, pady=6)
        ttk.Label(self.finished_frame, text='完成队列').pack(anchor='nw')
        # 使用 Treeview 显示多列
        self.finished_tree = ttk.Treeview(self.finished_frame, columns=('pid','arr','srv','fin','tat','wtat'), show='headings', height=20)
        self.finished_tree.heading('pid', text='标识')
        self.finished_tree.heading('arr', text='到达')
        self.finished_tree.heading('srv', text='服务')
        self.finished_tree.heading('fin', text='完成')
        self.finished_tree.heading('tat', text='周转')
        self.finished_tree.heading('wtat', text='带权周转')
        # 列宽和对齐
        self.finished_tree.column('pid', width=60, anchor='center')
        self.finished_tree.column('arr', width=50, anchor='center')
        self.finished_tree.column('srv', width=50, anchor='center')
        self.finished_tree.column('fin', width=70, anchor='center')
        self.finished_tree.column('tat', width=60, anchor='center')
        self.finished_tree.column('wtat', width=80, anchor='center')
        # 添加垂直滚动条
        self.finished_scroll = ttk.Scrollbar(self.finished_frame, orient='vertical', command=self.finished_tree.yview)
        self.finished_tree.configure(yscrollcommand=self.finished_scroll.set)
        self.finished_tree.pack(side='left', fill='y', pady=4)
        self.finished_scroll.pack(side='right', fill='y')

    def add_job(self):
        pid = self.entry_id.get().strip()
        arr = self.entry_arr.get().strip()
        srv = self.entry_srv.get().strip()
        if not pid or not arr.isdigit() or not srv.isdigit():
            messagebox.showwarning('输入错误', '请填写有效的 ID, 到达时间(整数), 服务时间(整数)')
            return
        proc = Process(pid, int(arr), int(srv))
        self.proc_list.append(proc)
        self.proc_list.sort(key=lambda p:p.arrive)
        # 为新进程分配颜色（若尚未分配）
        if pid not in self.color_map:
            idx = len(self.color_map) % self.cmap.N
            rgba = self.cmap(idx)
            # 转为 hex 便于 matplotlib 使用
            self.color_map[pid] = mcolors.to_hex(rgba)
        messagebox.showinfo('已添加', f'进程 {pid} 已添加')
        self.entry_id.delete(0,'end'); self.entry_arr.delete(0,'end'); self.entry_srv.delete(0,'end')

    def start(self):
        if self.running:
            return
        # update timeslices
        for i in range(self.queue_num):
            self.timeslice[i] = max(1, int(self.ts_vars[i].get()))
        self.running = True
        self.pause = False
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def toggle_pause(self):
        if not self.running:
            return
        self.pause = not self.pause

    def reset(self):
        with self.lock:
            self.running = False
            self.pause = False
            self.time = 0
            self.proc_list.clear()
            self.ready = [deque() for _ in range(self.queue_num)]
            self.current = None
            self.current_level = None
            self.current_slice_remain = 0
            self.gantt.clear()
            # clear finished list
            try:
                self.finished.clear()
                try:
                    # clear Treeview items
                    for iid in self.finished_tree.get_children():
                        self.finished_tree.delete(iid)
                except Exception:
                    pass
            except Exception:
                pass
            self._refresh_ui()
            self.ax.clear()
            self.ax.set_title('甘特图')
            self.canvas.draw()

    def export_gantt(self):
        # draw full gantt and save
        self._draw_gantt(final=True)
        fname = 'gantt_export.png'
        self.fig.savefig(fname, dpi=150)
        messagebox.showinfo('导出完成', f'甘特图已保存为 {fname}')

    def import_example(self):
        """导入一组示例进程到待到达队列（只能在仿真停止时）"""
        if self.running:
            messagebox.showwarning('禁止操作', '请先停止仿真后再导入示例')
            return
        # 示例集（可按需扩展或替换为从文件读取）
        # 使用与 add_job 相同的格式 (ID, 到达时间, 服务时间)，这里用数字 ID 示例：
        examples = [
            (0, 0, 20),
            (1, 3, 3),
            (2, 10, 23),
            (3, 20, 12),
            (4, 15, 9),
        ]
        with self.lock:
            for pid, arr, srv in examples:
                # 统一使用与“添加进程”相同的格式与校验：ID, 到达时间(整数), 服务时间(整数)
                pid_str = str(pid)
                arr_str = str(arr)
                srv_str = str(srv)
                if not arr_str.isdigit() or not srv_str.isdigit():
                    # 跳过不合法的示例行
                    continue
                # 避免 PID 冲突：若已有相同 pid，则在后面加索引
                base = pid_str
                unique_pid = pid_str
                idx = 1
                while any(p.pid == unique_pid for p in self.proc_list) or any(p.pid == unique_pid for q in self.ready for p in q) or (self.current and self.current.pid == unique_pid):
                    unique_pid = f"{base}{idx}"
                    idx += 1
                proc = Process(unique_pid, int(arr_str), int(srv_str))
                self.proc_list.append(proc)
                # 分配颜色（与 add_job 行为一致）
                if unique_pid not in self.color_map:
                    ci = len(self.color_map) % self.cmap.N
                    rgba = self.cmap(ci)
                    self.color_map[unique_pid] = mcolors.to_hex(rgba)
            # 保证按到达时间排序
            self.proc_list.sort(key=lambda p: p.arrive)
        messagebox.showinfo('导入完成', f'已导入 {len(examples)} 个示例进程')
        # 刷新界面（就绪队列及甘特）
        self._refresh_ui()

    def _run(self):
        # prepare lists
        with self.lock:
            # copy proc_list to arrival queue
            self.proc_list.sort(key=lambda p:p.arrive)
        while self.running:
            time.sleep(0.8)  # control speed for visualization
            if self.pause:
                continue
            with self.lock:
                # advance time
                # 加入到达作业
                arrivals = [p for p in self.proc_list if p.arrive <= self.time]
                for p in arrivals:
                    # ensure color assigned for this pid (in case proc was added programmatically)
                    if p.pid not in self.color_map:
                        idx = len(self.color_map) % self.cmap.N
                        rgba = self.cmap(idx)
                        self.color_map[p.pid] = mcolors.to_hex(rgba)
                    self.ready[0].append(Process(p.pid, p.arrive, p.service))
                    self.proc_list.remove(p)
                # 若当前没有运行，选择下一个
                if not self.current:
                    self._pick_next()
                else:
                    # 如果有新到达更高优先级且可剥夺，则立即抢占
                    if self.preemptible.get():
                        # highest non-empty queue index
                        higher_exists = False
                        for q in range(0, self.current_level):
                            if len(self.ready[q])>0:
                                higher_exists = True; break
                        if higher_exists:
                            # preempt current: put it back to the TAIL of its CURRENT queue
                            # and record its remaining slice so that when it resumes it continues the same slice
                            if self.current.remain>0:
                                # preserve remaining slice on the process
                                self.current.slice_remain = self.current_slice_remain
                                self.ready[self.current_level].append(self.current)
                            self.current = None
                            self.current_level = None
                            self.current_slice_remain = 0
                            self._pick_next()
                    # otherwise execute one time unit
                # execute 1 unit
                if self.current:
                    self._execute_unit()
                # refresh ui and plot
                self._refresh_ui()
                # stop condition
                if (not any(self.ready[i] for i in range(self.queue_num))) and (not self.current) and (not self.proc_list):
                    self.running = False
                    self._draw_gantt(final=True)
                    break
                self.time += 1

    def _pick_next(self):
        # pick from highest priority non-empty queue
        for i in range(self.queue_num):
            if self.ready[i]:
                proc = self.ready[i].popleft()
                self.current = proc
                self.current_level = i
                # 如果作业在被剥夺时保存了剩余时间片，则继续该剩余时间片；否则分配新的时间片
                if proc.slice_remain is None or proc.slice_remain <= 0:
                    proc.slice_remain = self.timeslice[i]
                self.current_slice_remain = proc.slice_remain
                # record segment start
                if proc.start_time is None:
                    proc.start_time = self.time
                # append gantt segment (we will merge consecutive same-process segments when drawing)
                self.gantt.append([proc.pid, self.time, 0, i])
                self.cur_var.set(f'当前: {proc.pid} (队列{i}) 剩余:{proc.remain}')
                return
        # nothing to run
        self.current = None
        self.current_level = None
        self.current_slice_remain = 0
        self.cur_var.set('当前: 空闲')

    def _execute_unit(self):
        # run one time unit
        self.current.remain -= 1
        self.current_slice_remain -= 1
        # also update the job's stored slice_remain (so if preempted it knows the remaining slice)
        self.current.slice_remain = self.current_slice_remain
        # extend last gantt segment duration
        if self.gantt:
            self.gantt[-1][2] += 1
        # if job finished
        if self.current.remain <= 0:
            # mark finish and move to finished list
            self.current.finish_time = self.time + 1
            self.current.slice_remain = None
            # append to finished list for display
            self.finished.append(self.current)
            # clear current
            self.current = None
            self.current_level = None
            self.current_slice_remain = 0
        else:
            # slice used up -> demote
            if self.current_slice_remain <= 0:
                target = min(self.current_level+1, self.queue_num-1)
                # when demoting, reset stored slice_remain so next time it gets a full new timeslice
                self.current.slice_remain = None
                self.ready[target].append(self.current)
                self.current = None
                self.current_level = None
                self.current_slice_remain = 0

    def _refresh_ui(self):
        # update listboxes
        for i in range(self.queue_num):
            self.queue_listboxes[i].delete(0,'end')
            for proc in self.ready[i]:
                self.queue_listboxes[i].insert('end', f'{proc.pid} (rem={proc.remain})')
            # also show current if level matches
            if self.current and self.current_level==i:
                self.queue_listboxes[i].insert('end', f'-- RUNNING: {self.current.pid} (rem={self.current.remain})')
        self.time_var.set(f'时间: {self.time}')
        if not self.running:
            self.time_var.set(f'时间: {self.time} (已停止)')
        # update finished table (右侧) — 显示 PID, 到达, 服务, 完成, 周转, 带权周转
        try:
            # clear existing rows
            for iid in self.finished_tree.get_children():
                self.finished_tree.delete(iid)
            for p in self.finished:
                arr = getattr(p, 'arrive', '')
                srv = getattr(p, 'service', '')
                fin = getattr(p, 'finish_time', '')
                if fin != '' and fin is not None:
                    tat = fin - arr
                    wtat = tat / srv if srv and srv>0 else 0
                    wtat_str = f"{wtat:.2f}"
                else:
                    tat = ''
                    wtat_str = ''
                # insert into Treeview
                try:
                    self.finished_tree.insert('', 'end', values=(str(p.pid), str(arr), str(srv), str(fin), str(tat), wtat_str))
                except Exception:
                    # fallback: skip if tree not available
                    pass
        except Exception:
            pass
        # redraw gantt
        self._draw_gantt()

    def _draw_gantt(self, final=False):
        self.ax.clear()
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
        y_height = 8
        for seg in self.gantt:
            pid, s, dur, level = seg
            if dur<=0: continue
            y = (self.queue_num - 1 - level)*10
            color = self.color_map.get(pid, colors[hash(pid)%len(colors)])
            self.ax.broken_barh([(s,dur)], (y, y_height), facecolors=color)
            self.ax.text(s + dur/2, y + y_height/2, pid, ha='center', va='center', color='white', fontsize=8)
        self.ax.set_ylim(0, self.queue_num*10)
        self.ax.set_xlim(0, max(self.time+5, max((s+dur) for (_,s,dur,_) in self.gantt) if self.gantt else 10))
        self.ax.set_yticks([])
        self.ax.set_xlabel('时间')
        self.ax.set_title('甘特图' + (' (已结束)' if final else ''))
        # 绘制进程颜色图例（基于当前 color_map），放在图右侧并随重绘更新
        try:
            handles = []
            labels = []
            # 按 PID 排序以使图例顺序稳定
            for pid in sorted(self.color_map.keys()):
                col = self.color_map[pid]
                handles.append(mpatches.Patch(facecolor=col, edgecolor='k'))
                labels.append(str(pid))
            # 先移除旧图例（若存在）以避免重复
            if self.ax.get_legend():
                self.ax.get_legend().remove()
            if handles:
                self.ax.legend(handles, labels, title='进程', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        except Exception:
            pass
        self.canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = MFQApp(root)
    root.mainloop()
