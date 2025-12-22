import importlib.util
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from page import SimplePager
from pv import CountingSemaphore


STATE_NEW = "新建"
STATE_READY = "就绪"
STATE_RUNNING = "运行"
STATE_BLOCKED = "阻塞"
STATE_SUSP_READY = "挂起就绪"
STATE_SUSP_BLOCKED = "挂起阻塞"
STATE_EXIT = "完成"
ALL_STATES = [
    STATE_NEW,
    STATE_READY,
    STATE_RUNNING,
    STATE_BLOCKED,
    STATE_SUSP_READY,
    STATE_SUSP_BLOCKED,
    STATE_EXIT,
]


class Job:
    def __init__(self, jid, arrive, service, pages):
        self.jid = str(jid)
        self.arrive = int(arrive)
        self.service = int(service)
        self.remaining = int(service)
        self.pages = int(pages)
        self.state = STATE_NEW
        self.current_page = 0
        self.log = []

    def tick(self):
        if self.remaining > 0:
            self.remaining -= 1
        self.current_page = (self.current_page + 1) % max(1, self.pages)


class KernelSimulator(tk.Toplevel):
    """Seven-state pipeline that orchestrates scheduling, paging, VM and IO."""

    def __init__(self, master=None):
        super().__init__(master)
        self.title("作业七态 & 模块通信")
        self.geometry("1080x720")

        # Modules reused from repo
        self.bitmap_pager = SimplePager(frame_count=24)  # 位示图分页
        self.vm_pager = self._load_vm_pager()
        self.io_slots = CountingSemaphore(2)  # 模拟外设并发度（PV）

        self.time = 0
        self.jobs = []
        self.ready_queue = []
        self.running = None
        self.blocked = []
        self.suspended_ready = []
        self.suspended_blocked = []
        self.finished = []
        self.auto = False
        self.after_id = None

        self._build_ui()

    def _load_vm_pager(self):
        vm_path = Path(__file__).resolve().parent.parent / "virtual memory.py"
        spec = importlib.util.spec_from_file_location("virtual_memory_app", vm_path)
        if spec is None or spec.loader is None:
            raise ImportError("无法加载虚拟存储器模块")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DemandPager(frame_count=24, per_proc_alloc=3)

    # UI -----------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="到达时间").grid(row=0, column=0)
        self.ent_arrive = ttk.Entry(top, width=6)
        self.ent_arrive.insert(0, "0")
        self.ent_arrive.grid(row=0, column=1)
        ttk.Label(top, text="服务时间").grid(row=0, column=2)
        self.ent_service = ttk.Entry(top, width=6)
        self.ent_service.insert(0, "8")
        self.ent_service.grid(row=0, column=3)
        ttk.Label(top, text="页数").grid(row=0, column=4)
        self.ent_pages = ttk.Entry(top, width=6)
        self.ent_pages.insert(0, "6")
        self.ent_pages.grid(row=0, column=5)
        ttk.Button(top, text="添加作业", command=self.add_job).grid(row=0, column=6, padx=6)
        ttk.Button(top, text="导入示例", command=self.load_examples).grid(row=0, column=7, padx=6)

        ttk.Button(top, text="单步", command=self.step).grid(row=0, column=8, padx=6)
        self.btn_auto = ttk.Button(top, text="开始自动", command=self.toggle_auto)
        self.btn_auto.grid(row=0, column=9, padx=6)
        ttk.Button(top, text="重置", command=self.reset).grid(row=0, column=10, padx=6)

        self.status = tk.StringVar(value="时间 0")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=10)

        # state panels
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=6)

        self.listboxes = {}
        for i, state in enumerate(ALL_STATES):
            frame = ttk.LabelFrame(body, text=state)
            frame.grid(row=i // 3, column=i % 3, padx=8, pady=6, sticky="nsew")
            lst = tk.Listbox(frame, width=28, height=9)
            lst.pack(fill="both", expand=True)
            self.listboxes[state] = lst
            body.columnconfigure(i % 3, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(2, weight=1)

        # logs and explanations
        self.log_text = tk.Text(self, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.log_text.insert(
            "end",
            "作业会依次经历：新建 -> 就绪 -> 运行 -> 阻塞/挂起 -> 完成。\n"
            "调度：就绪队列简单轮转；运行时访问页号，缺页通过请求分页装入；\n"
            "若位示图分页页框不足则挂起；IO 模拟使用 PV 信号量控制阻塞/唤醒。\n",
        )
        self.log_text.configure(state="disabled")

    # Job operations ------------------------------------------------------
    def add_job(self):
        try:
            arrive = int(self.ent_arrive.get())
            service = int(self.ent_service.get())
            pages = int(self.ent_pages.get())
        except ValueError:
            messagebox.showwarning("输入错误", "请填写整数到达、服务时间和页数")
            return
        jid = len(self.jobs) + 1
        job = Job(jid, arrive, service, pages)
        self.jobs.append(job)
        self._log(f"作业 {job.jid} 创建，等待到达 {arrive}")
        self._refresh()

    def load_examples(self):
        samples = [(0, 9, 6), (2, 6, 4), (4, 10, 8)]
        for arr, srv, pages in samples:
            jid = len(self.jobs) + 1
            self.jobs.append(Job(jid, arr, srv, pages))
        self._log("已导入示例作业，时间到达后进入就绪")
        self._refresh()

    def reset(self):
        if self.after_id:
            self.after_cancel(self.after_id)
        self.time = 0
        self.jobs.clear()
        self.ready_queue.clear()
        self.blocked.clear()
        self.suspended_ready.clear()
        self.suspended_blocked.clear()
        self.finished.clear()
        self.running = None
        self.auto = False
        self.btn_auto.config(text="开始自动")
        self.bitmap_pager = SimplePager(frame_count=24)
        self.vm_pager = self._load_vm_pager()
        self.io_slots = CountingSemaphore(2)
        self._refresh()
        self._log("已重置")

    # Simulation ---------------------------------------------------------
    def toggle_auto(self):
        self.auto = not self.auto
        self.btn_auto.config(text="停止自动" if self.auto else "开始自动")
        if self.auto:
            self._auto_step()

    def _auto_step(self):
        self.step()
        if self.auto:
            self.after_id = self.after(800, self._auto_step)

    def step(self):
        # move arrivals
        for job in self.jobs:
            if job.state == STATE_NEW and job.arrive <= self.time:
                self._register_job(job)
                job.state = STATE_READY
                self.ready_queue.append(job)
                self._log(f"作业 {job.jid} 到达 -> 就绪")

        # dispatch if idle
        if not self.running and self.ready_queue:
            self.running = self.ready_queue.pop(0)
            self.running.state = STATE_RUNNING
            self._log(f"调度 -> 运行 {self.running.jid}")

        # run one tick
        if self.running:
            self._execute_running(self.running)

        # IO unblocks
        self._service_io()

        self.time += 1
        self.status.set(f"时间 {self.time} | 就绪 {len(self.ready_queue)} 运行 {self.running.jid if self.running else '-'}")
        self._refresh()

    def _execute_running(self, job: Job):
        # ensure pages loaded via bitmap pager; if不足挂起
        allocated = self.bitmap_pager.allocate_pages_first_fit(job.jid) if job.pages else 0
        if allocated < job.pages and job.state == STATE_RUNNING:
            job.state = STATE_SUSP_READY
            self.suspended_ready.append(job)
            self.running = None
            self._log(f"作业 {job.jid} 页面不足，挂起等待内存")
            return

        # access one page through demand paging
        page = job.current_page
        try:
            res = self.vm_pager.access_page(job.jid, page, is_write=True)
            if res.get("hit"):
                self._log(f"{job.jid} 访问页 {page} 命中帧 {res['frame']}")
            else:
                ev = res.get("evicted")
                if ev:
                    self._log(f"{job.jid} 缺页装入帧 {res['frame']} 置换 {ev[0]}:{ev[1]}")
                else:
                    self._log(f"{job.jid} 缺页装入帧 {res['frame']}")
        except ValueError as exc:
            job.state = STATE_SUSP_READY
            self.suspended_ready.append(job)
            self.running = None
            self._log(f"{job.jid} 请求分页失败: {exc} -> 挂起就绪")
            return

        # random IO triggers PV semaphore
        if random.random() < 0.25:
            if self.io_slots.value() > 0:
                self.io_slots.acquire()
                job.state = STATE_BLOCKED
                self.blocked.append(job)
                self._log(f"{job.jid} 发起 IO -> 阻塞")
                self.running = None
                return

        job.tick()
        if job.remaining <= 0:
            job.state = STATE_EXIT
            self.finished.append(job)
            self._release_job(job)
            self._log(f"作业 {job.jid} 完成")
            self.running = None
        else:
            # round robin
            job.state = STATE_READY
            self.ready_queue.append(job)
            self.running = None

    def _service_io(self):
        if not self.blocked:
            return
        wake_count = min(len(self.blocked), 2)
        for _ in range(wake_count):
            job = self.blocked.pop(0)
            try:
                self.io_slots.release()
            except Exception:
                pass
            # 如果内存仍不足，先进入挂起阻塞
            if self.bitmap_pager.free_count() < job.pages:
                job.state = STATE_SUSP_BLOCKED
                self.suspended_blocked.append(job)
                self._log(f"IO 完成但内存不足，作业 {job.jid} -> 挂起阻塞")
            else:
                job.state = STATE_READY
                self.ready_queue.append(job)
                self._log(f"IO 完成，作业 {job.jid} 唤醒 -> 就绪")

        # wake suspended when frames free
        freed = []
        for job in list(self.suspended_ready):
            if self.bitmap_pager.free_count() >= job.pages:
                self.bitmap_pager.allocate_pages_first_fit(job.jid)
                job.state = STATE_READY
                self.ready_queue.append(job)
                freed.append(job)
                self._log(f"内存腾挪，作业 {job.jid} 从挂起就绪恢复")
        for job in freed:
            self.suspended_ready.remove(job)

        freed_block = []
        for job in list(self.suspended_blocked):
            if self.bitmap_pager.free_count() >= job.pages:
                self.bitmap_pager.allocate_pages_first_fit(job.jid)
                job.state = STATE_READY
                self.ready_queue.append(job)
                freed_block.append(job)
                self._log(f"内存腾挪，作业 {job.jid} 从挂起阻塞恢复 -> 就绪")
        for job in freed_block:
            self.suspended_blocked.remove(job)

    # helpers ------------------------------------------------------------
    def _register_job(self, job: Job):
        pid = job.jid
        try:
            self.bitmap_pager.add_process(pid, job.pages)
        except ValueError:
            pass
        try:
            self.vm_pager.add_process(pid, job.pages)
        except ValueError:
            pass

    def _release_job(self, job: Job):
        pid = job.jid
        try:
            self.bitmap_pager.deallocate_process(pid)
        except Exception:
            pass
        try:
            self.vm_pager.deallocate_process(pid)
        except Exception:
            pass

    def _refresh(self):
        for state, lst in self.listboxes.items():
            lst.delete(0, "end")
        for job in self.jobs:
            self.listboxes[job.state].insert("end", f"J{job.jid} rem={job.remaining} 页={job.pages}")
        for job in self.finished:
            if job not in self.jobs:
                self.listboxes[STATE_EXIT].insert("end", f"J{job.jid}")

    def _log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[t={self.time}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def run():
    root = tk.Tk()
    root.withdraw()
    KernelSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    run()
