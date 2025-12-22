import tkinter as tk
from tkinter import ttk
import random
import time
from collections import deque


class VMProcess:
    def __init__(self, pid, num_pages):
        self.pid = str(pid)
        self.num_pages = int(num_pages)
        # 页表：page -> {'frame': index or -1, 'dirty': bool}
        self.page_table = {i: {'frame': -1, 'dirty': False} for i in range(self.num_pages)}
        # 当前驻留的页数（避免频繁全表统计）
        self.resident_count = 0
        # 固定分配给该进程的物理帧索引列表
        self.allocated_frames = []
        # 进程内的 FIFO 置换队列（存帧号）
        self.frame_fifo = deque()
        # 自动模式下待访问的页序列
        self.todo_pages = deque()


class DemandPager:
    def __init__(self, frame_count=20, per_proc_alloc=3):
        self.frame_count = frame_count
        self.frames = [None] * frame_count  # None or (pid, page)
        self.frame_owner = [None] * frame_count  # 记录帧所属进程（固定分配）
        self.processes = {}  # pid -> VMProcess
        self.process_fifo = deque()  # process arrival order
        # 磁盘槽：list of {'pid','page','in_memory'}，None 表示空闲槽
        self.disk_slots = []
        self.free_disk_slots = deque()
        # 每个进程固定分配的页框数
        self.per_proc_alloc = 3

    def set_frames(self, n):
        self.frame_count = n
        self.frames = [None] * n
        self.frame_owner = [None] * n
        self.process_fifo.clear()
        for slot in self.disk_slots:
            if slot is not None:
                slot['in_memory'] = False
        for p in self.processes.values():
            p.page_table = {i: {'frame': -1, 'dirty': False} for i in range(p.num_pages)}
            p.resident_count = 0
            p.allocated_frames = []
            p.frame_fifo.clear()
            p.todo_pages = deque()

    def add_process(self, pid, num_pages):
        if pid in self.processes:
            raise ValueError('PID 已存在')
        p = VMProcess(pid, num_pages)
        # 固定分配：每进程最多 3 帧，若进程页数更少则按页数分配，仍受可用空闲帧限制
        free_frames = [i for i, owner in enumerate(self.frame_owner) if owner is None]
        free_cnt = len(free_frames)
        if free_cnt == 0:
            raise ValueError('可用物理帧不足，无法分配')
        target = min(self.per_proc_alloc, num_pages, free_cnt)
        p.allocated_frames = free_frames[:target]
        for fidx in p.allocated_frames:
            self.frame_owner[fidx] = p.pid
        p.frame_fifo = deque()
        p.todo_pages = deque(range(p.num_pages))
        self.processes[pid] = p
        self.process_fifo.append(pid)
        # 为该进程的每一页在磁盘上分配一个槽（复用已释放槽）
        p.disk_table = {}
        for page in range(p.num_pages):
            if self.free_disk_slots:
                didx = self.free_disk_slots.popleft()
                self.disk_slots[didx] = {'pid': pid, 'page': page, 'in_memory': False}
            else:
                didx = len(self.disk_slots)
                self.disk_slots.append({'pid': pid, 'page': page, 'in_memory': False})
            p.disk_table[page] = didx
        # 每个进程的页框配额（不超过该进程页数）
        p.max_frames = min(p.num_pages, max(1, self.per_proc_alloc))
        return p

    def deallocate_process(self, pid):
        if pid not in self.processes:
            return 0
        p = self.processes[pid]
        freed = 0
        for page, entry in list(p.page_table.items()):
            frame = entry.get('frame', -1)
            if frame is not None and frame != -1 and 0 <= frame < self.frame_count:
                cur = self.frames[frame]
                if cur and cur[0] == pid and cur[1] == page:
                    self.frames[frame] = None
                    freed += 1
                    if p.resident_count > 0:
                        p.resident_count -= 1
                    entry['frame'] = -1
                    entry['dirty'] = False
        # 释放该进程在磁盘上的槽，加入空闲槽池
        for page, didx in getattr(p, 'disk_table', {}).items():
            if 0 <= didx < len(self.disk_slots):
                self.disk_slots[didx] = None
                self.free_disk_slots.append(didx)

        # 释放固定帧占用
        for fidx in p.allocated_frames:
            if 0 <= fidx < len(self.frame_owner):
                if self.frames[fidx] is None or (self.frames[fidx] and self.frames[fidx][0] == pid):
                    self.frames[fidx] = None
                self.frame_owner[fidx] = None

        del self.processes[pid]
        try:
            self.process_fifo.remove(pid)
        except ValueError:
            pass
        return freed

    def access_page(self, pid, page, is_write=True):
        """访问/写入一个虚拟页；固定分配：仅在进程分配的帧中置换。"""
        if pid not in self.processes:
            raise ValueError('未知 PID')
        p = self.processes[pid]
        if page < 0 or page >= p.num_pages:
            raise ValueError('页号越界')

        if not p.allocated_frames:
            raise ValueError('该进程无可用分配帧')

        entry = p.page_table.get(page, {'frame': -1, 'dirty': False})
        if entry.get('frame', -1) != -1:
            # hit
            if is_write:
                entry['dirty'] = True
            return {'hit': True, 'frame': entry['frame'], 'dirty': entry['dirty'], 'disk': p.disk_table.get(page)}

        # page fault: need to load page (固定分配范围内寻找空帧或本进程内部置换)
        target_frame = None
        for fidx in p.allocated_frames:
            if self.frames[fidx] is None:
                target_frame = fidx
                break

        evicted = None
        if target_frame is None and p.frame_fifo:
            target_frame = p.frame_fifo.popleft()
            old = self.frames[target_frame]
            if old:
                old_pid, old_page = old
                evicted = (old_pid, old_page, target_frame)
                old_proc = self.processes.get(old_pid)
                if old_proc:
                    old_entry = old_proc.page_table.get(old_page, {'frame': -1, 'dirty': False})
                    if old_entry.get('dirty'):
                        didx_old = getattr(old_proc, 'disk_table', {}).get(old_page)
                        if didx_old is not None and 0 <= didx_old < len(self.disk_slots):
                            slot = self.disk_slots[didx_old]
                            if slot is not None:
                                slot['in_memory'] = False
                    if old_proc.resident_count > 0:
                        old_proc.resident_count -= 1
                    old_entry['frame'] = -1
                    old_entry['dirty'] = False
                    if target_frame in old_proc.frame_fifo:
                        try:
                            old_proc.frame_fifo.remove(target_frame)
                        except ValueError:
                            pass

        if target_frame is None:
            return {'hit': False, 'frame': -1, 'evicted': None}

        self.frames[target_frame] = (pid, page)
        entry['frame'] = target_frame
        entry['dirty'] = bool(is_write)
        p.page_table[page] = entry
        p.resident_count += 1
        # 进程内部 FIFO
        try:
            p.frame_fifo.remove(target_frame)
        except ValueError:
            pass
        p.frame_fifo.append(target_frame)
        didx = getattr(p, 'disk_table', {}).get(page)
        if didx is not None and 0 <= didx < len(self.disk_slots):
            slot = self.disk_slots[didx]
            if slot is not None:
                slot['in_memory'] = True
        return {'hit': False, 'frame': target_frame, 'evicted': evicted, 'disk': didx, 'dirty': entry['dirty']}


class VMApp:
    def __init__(self, root):
        self.root = root
        self.root.title('基本虚拟存储器管理（请求分页可视化）')
        self.pager = DemandPager(frame_count=20, per_proc_alloc=2)
        self._build_ui()
        self.color_map = {}
        self.color_palette = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#CC99FF', '#FFFF99', '#FF99CC', '#CCE5FF']
        self.auto_running = False
        self.auto_after_id = None
        self.next_auto_pid = 10
        self._refresh()
        # 自动加入示例并运行
        self.create_sample()

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill='x', padx=6, pady=6)

        ttk.Label(top, text='内存页框数:').grid(row=0, column=0)
        self.frame_var = tk.IntVar(value=self.pager.frame_count)
        ttk.Spinbox(top, from_=1, to=128, textvariable=self.frame_var, width=6).grid(row=0, column=1)
        ttk.Button(top, text='应用页框数', command=self.apply_frames).grid(row=0, column=2, padx=6)

        ttk.Label(top, text='每进程页框配额:').grid(row=0, column=12)
        self.per_alloc_var = tk.IntVar(value=self.pager.per_proc_alloc)
        ttk.Spinbox(top, from_=1, to=32, textvariable=self.per_alloc_var, width=6).grid(row=0, column=13)
        ttk.Button(top, text='应用配额', command=self.apply_per_alloc).grid(row=0, column=14, padx=6)

        ttk.Label(top, text='进程 ID').grid(row=0, column=3)
        self.pid_entry = ttk.Entry(top, width=8)
        self.pid_entry.grid(row=0, column=4)
        ttk.Label(top, text='页数').grid(row=0, column=5)
        self.pages_entry = ttk.Entry(top, width=6)
        self.pages_entry.grid(row=0, column=6)
        ttk.Button(top, text='创建进程', command=self.create_process).grid(row=0, column=7, padx=6)
        ttk.Button(top, text='释放进程', command=self.release_selected).grid(row=0, column=8, padx=6)
        ttk.Button(top, text='访问页', command=self.access_from_inputs).grid(row=0, column=9, padx=6)
        self.auto_btn = ttk.Button(top, text='开始自动', command=self.toggle_auto)
        self.auto_btn.grid(row=0, column=10, padx=6)

        mid = ttk.Frame(self.root)
        mid.pack(fill='both', expand=1, padx=6, pady=6)

        # canvas for frames (左)
        frame_box = ttk.LabelFrame(mid, text='内存页框')
        frame_box.pack(side='left', fill='both', expand=1, padx=6, pady=6)
        self.canvas = tk.Canvas(frame_box, width=420, height=300, bg='white')
        self.canvas.pack(fill='both', expand=1)

        # middle: 页表
        middle = ttk.Frame(mid)
        middle.pack(side='left', fill='y', padx=6)
        ttk.Label(middle, text='页表').pack()
        self.pt_tree = ttk.Treeview(middle, columns=('page','loc','dirty'), show='headings', height=20)
        self.pt_tree.heading('page', text='页号')
        self.pt_tree.heading('loc', text='位置')
        self.pt_tree.heading('dirty', text='脏')
        self.pt_tree.column('page', width=40, anchor='center')
        self.pt_tree.column('loc', width=100, anchor='center')
        self.pt_tree.column('dirty', width=40, anchor='center')
        self.pt_tree.pack(fill='both', expand=1)
        self.alloc_var = tk.StringVar(value='分配帧：-')
        ttk.Label(middle, textvariable=self.alloc_var).pack(fill='x', pady=(4,0))

        # right: 磁盘视图
        right = ttk.LabelFrame(mid, text='磁盘（虚存）')
        right.pack(side='left', fill='both', expand=1, padx=6)
        disk_wrap = ttk.Frame(right)
        disk_wrap.pack(fill='both', expand=1)
        self.disk_canvas = tk.Canvas(disk_wrap, width=320, height=320, bg='#f7f7f7')
        v_scroll = ttk.Scrollbar(disk_wrap, orient='vertical', command=self.disk_canvas.yview)
        h_scroll = ttk.Scrollbar(disk_wrap, orient='horizontal', command=self.disk_canvas.xview)
        self.disk_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.disk_canvas.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        disk_wrap.rowconfigure(0, weight=1)
        disk_wrap.columnconfigure(0, weight=1)
        ttk.Label(right, text='进程').pack()
        self.proc_list = tk.Listbox(right, width=12, height=20)
        self.proc_list.pack(fill='y')
        self.proc_list.bind('<<ListboxSelect>>', self.show_page_table)

        # 右侧不再需要原来的页表 tree

        # status
        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(self.root, textvariable=self.status_var).pack(fill='x')
        # log panel
        log_box = ttk.LabelFrame(self.root, text='操作日志')
        log_box.pack(fill='both', padx=6, pady=(0,6))
        self.log_text = tk.Text(log_box, height=6, wrap='word', state='disabled')
        self.log_text.pack(fill='both', expand=1)

    def apply_frames(self):
        n = int(self.frame_var.get())
        self.pager.set_frames(n)
        self._log(f'应用内存页框数 {n}')
        self._refresh()

    def apply_per_alloc(self):
        # 固定分配 3 帧；保持 UI 同步显示
        self.pager.per_proc_alloc = 3
        self.per_alloc_var.set(3)
        self._log('固定分配每进程 3 帧')
        self._refresh()

    def create_process(self):
        pid = self.pid_entry.get().strip()
        pages = self.pages_entry.get().strip()
        if not pid.isdigit() or not pages.isdigit():
            self.status_var.set('请输入数字 PID 和 页数')
            return
        try:
            self.pager.add_process(pid, int(pages))
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self._log(f'创建进程 {pid}（{pages} 页）')
        self.pid_entry.delete(0,'end'); self.pages_entry.delete(0,'end')
        self._refresh()

    def release_selected(self):
        sel = self.proc_list.curselection()
        if not sel:
            return
        pid = self.proc_list.get(sel[0])
        self.pager.deallocate_process(pid)
        # no popup on release
        self._log(f'释放进程 {pid}')
        self._refresh()

    def access_from_inputs(self):
        sel = self.proc_list.curselection()
        if not sel:
            self.status_var.set('请选择进程或输入 PID')
            return
        pid = self.proc_list.get(sel[0])
        # ask for page via simple dialog using pages_entry
        page_s = self.pages_entry.get().strip()
        if not page_s.isdigit():
            self.status_var.set('请在 页数 输入要访问的页号')
            return
        page = int(page_s)
        res = self.pager.access_page(pid, page, is_write=True)
        if res['hit']:
            self.status_var.set(f'命中: PID {pid} 页 {page} 在帧 {res["frame"]}')
        else:
            if res.get('evicted'):
                old_pid, old_page, idx = res['evicted']
                self.status_var.set(f'缺页, 载入到帧 {res["frame"]}, 置换 {old_pid}:{old_page}')
            else:
                self.status_var.set(f'缺页, 载入到帧 {res["frame"]}')
        self._log_access(pid, page, res, is_write=True)
        self._refresh()

    def _get_color(self, pid):
        if not pid:
            return '#FFFFFF'
        if pid not in self.color_map:
            self.color_map[pid] = self.color_palette[len(self.color_map) % len(self.color_palette)]
        return self.color_map[pid]

    def _draw_frames(self):
        self.canvas.delete('all')
        cols = min(12, max(1, self.pager.frame_count))
        size = 44
        pad = 8
        for i in range(self.pager.frame_count):
            row = i // cols
            col = i % cols
            x0 = pad + col * (size + pad)
            y0 = pad + row * (size + pad)
            x1 = x0 + size
            y1 = y0 + size
            owner_tuple = self.pager.frames[i]
            reserved_pid = self.pager.frame_owner[i]
            if owner_tuple:
                pid, page = owner_tuple
                fill = self._get_color(pid)
                proc = self.pager.processes.get(pid)
                entry = proc.page_table.get(page, {}) if proc else {}
                dirty = entry.get('dirty')
                text = f'{pid}:{page}{"*" if dirty else ""}'
            elif reserved_pid:
                fill = self._get_color(reserved_pid)
                text = f'{reserved_pid}'
            else:
                fill = '#FFFFFF'
                text = str(i)
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline='#333')
            self.canvas.create_text((x0 + x1)/2, (y0 + y1)/2, text=text, font=('Arial', 10))

    def show_page_table(self, evt=None):
        for iid in self.pt_tree.get_children():
            self.pt_tree.delete(iid)
        sel = self.proc_list.curselection()
        if not sel:
            self.alloc_var.set('分配帧：-')
            return
        pid = self.proc_list.get(sel[0])
        p = self.pager.processes.get(pid)
        if not p:
            self.alloc_var.set('分配帧：-')
            return
        alloc_str = ','.join(str(i) for i in p.allocated_frames) if p.allocated_frames else '-'
        self.alloc_var.set(f'分配帧：{alloc_str}')
        for page in range(p.num_pages):
            entry = p.page_table.get(page, {'frame': -1, 'dirty': False})
            frame = entry.get('frame', -1)
            if frame is not None and frame != -1:
                loc = f'内存#{frame}'
            else:
                didx = getattr(p, 'disk_table', {}).get(page)
                loc = f'磁盘#{didx}' if didx is not None else '空'
            dirty_flag = '是' if entry.get('dirty') else '否'
            self.pt_tree.insert('', 'end', values=(page, loc, dirty_flag))

    def _draw_disk(self):
        self.disk_canvas.delete('all')
        # draw disk slots in rows
        total = len(self.pager.disk_slots)
        if total == 0:
            return
        cols = 6
        size = 36
        pad = 6
        for i, val in enumerate(self.pager.disk_slots):
            row = i // cols
            col = i % cols
            x0 = pad + col * (size + pad)
            y0 = pad + row * (size + pad)
            x1 = x0 + size
            y1 = y0 + size
            if val is None:
                # empty disk slot
                fill = '#FFFFFF'
                outline = '#CCC'
                text = str(i)
            else:
                pid = val.get('pid')
                page = val.get('page')
                in_mem = val.get('in_memory')
                fill = self._get_color(pid)
                outline = '#666' if not in_mem else '#333'
                text = f'{pid}:{page}{"*" if in_mem else ""}'
            self.disk_canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline)
            self.disk_canvas.create_text((x0 + x1)/2, (y0 + y1)/2, text=text, font=('Arial', 8))
        # 依据内容设置滚动区域
        rows = (total + cols - 1) // cols
        width = pad + cols * (size + pad)
        height = pad + rows * (size + pad)
        self.disk_canvas.configure(scrollregion=(0, 0, width, height))

    def _refresh(self):
        # redraw frames
        self._draw_frames()
        # refresh process list
        cur = self.proc_list.curselection()
        self.proc_list.delete(0,'end')
        for pid in list(self.pager.processes.keys()):
            self.proc_list.insert('end', pid)
        try:
            if cur:
                self.proc_list.select_set(cur[0])
        except Exception:
            pass
        self.show_page_table()
        # also redraw disk
        self._draw_disk()

    def toggle_auto(self):
        if self.auto_running:
            self.auto_running = False
            self.auto_btn.config(text='开始自动')
            if self.auto_after_id:
                try:
                    self.root.after_cancel(self.auto_after_id)
                except Exception:
                    pass
            self._log('自动运行停止')
        else:
            self.auto_running = True
            self.auto_btn.config(text='停止自动')
            self._log('自动运行开始')
            self._auto_step()

    def create_sample(self):
        # 自动加入样例进程并开始自动运行
        # 样例：页数更小，生命周期更短，便于频繁回收与再创建
        samples = [(1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 5)]
        for pid, pages in samples:
            try:
                self.pager.add_process(str(pid), pages)
            except ValueError:
                pass
        self._log('已加载示例进程并启动自动运行')
        self._refresh()
        # start auto
        if not self.auto_running:
            self.toggle_auto()

        # adjust next pid
        try:
            existing = [int(x) for x in self.pager.processes.keys() if x.isdigit()]
            if existing:
                self.next_auto_pid = max(existing) + 1
        except Exception:
            pass

    def _auto_step(self):
        if not self.auto_running:
            return
        # 轮询进程，逐页完成其生命周期；每个页访问一次后计为完成
        def _fill_free_frames():
            # 尽量用新进程填满所有空闲帧，保持内存高利用
            try_count = 0
            while True:
                free_frames = [i for i, owner in enumerate(self.pager.frame_owner) if owner is None]
                if not free_frames:
                    break
                new_pid = str(self.next_auto_pid)
                pages = random.randint(6, 12)
                try:
                    self.pager.add_process(new_pid, pages)
                    self.status_var.set(f'自动创建进程 {new_pid}（{pages} 页）')
                    self._log(f'自动创建进程 {new_pid}（{pages} 页）')
                    self.next_auto_pid += 1
                    self._refresh()
                except ValueError:
                    self.next_auto_pid += 1
                    try_count += 1
                    if try_count >= 3:
                        break

        while True:
            # 若无进程/队列为空，则尝试自动创建新进程保持运行
            if not self.pager.processes or not self.pager.process_fifo:
                _fill_free_frames()
                if not self.pager.processes or not self.pager.process_fifo:
                    self.status_var.set('无可分配帧，等待回收')
                    self.auto_after_id = self.root.after(700, self._auto_step)
                    return
                continue

            pid = self.pager.process_fifo[0]
            proc = self.pager.processes.get(pid)
            if not proc:
                # 可能已被手动释放
                self.pager.process_fifo.popleft()
                continue

            if not proc.todo_pages:
                self.status_var.set(f'进程 {pid} 完成，回收')
                self._log(f'进程 {pid} 完成，回收帧 {proc.allocated_frames}')
                self.pager.deallocate_process(pid)
                self._refresh()
                # 尝试立即补充新进程，保持持续运行
                _fill_free_frames()
                continue

            page = proc.todo_pages.popleft()
            is_write = random.random() < 0.6
            res = self.pager.access_page(pid, page, is_write=is_write)
            action = '写' if is_write else '读'
            if res['hit']:
                self.status_var.set(f'自动{action} 命中 {pid}:{page} 在帧 {res["frame"]}')
            else:
                if res.get('evicted'):
                    old_pid, old_page, idx = res['evicted']
                    self.status_var.set(f'自动{action} 缺页 {pid}:{page} -> 帧 {res["frame"]} 置换 {old_pid}:{old_page}')
                else:
                    self.status_var.set(f'自动{action} 缺页 {pid}:{page} -> 帧 {res["frame"]}')
            self._log_access(pid, page, res, is_write=is_write)
            # 将当前进程放到队尾，轮转执行
            self.pager.process_fifo.rotate(-1)
            break
        self._refresh()
        self.auto_after_id = self.root.after(700, self._auto_step)

    def _log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        line = f'[{ts}] {msg}\\n'
        self.log_text.configure(state='normal')
        self.log_text.insert('end', line)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _log_access(self, pid, page, res, is_write=True):
        # 单行记录：操作类型 + 结果 + 位置/置换/来源
        op = '写' if is_write else '读'
        parts = [f'{op} PID {pid} 页 {page}']
        if res.get('hit'):
            parts.append(f'命中帧 {res["frame"]}')
            if res.get('dirty'):
                parts.append('脏')
        else:
            parts.append(f'缺页 -> 帧 {res["frame"]}')
            if res.get('disk') is not None:
                parts.append(f'源槽#{res["disk"]}')
            evicted = res.get('evicted')
            if evicted:
                old_pid, old_page, _ = evicted
                parts.append(f'置换 {old_pid}:{old_page}')
            if res.get('dirty'):
                parts.append('脏')
        self._log(' | '.join(parts))


if __name__ == '__main__':
    root = tk.Tk()
    app = VMApp(root)
    root.geometry('1000x580')
    root.mainloop()
