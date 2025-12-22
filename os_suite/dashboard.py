import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from MFQ import MFQApp
from page import PageApp
from pv import PVApp
from .integration import KernelSimulator


def _load_virtual_memory_app():
    """Dynamically load VMApp from the existing `virtual memory.py` file."""
    vm_path = Path(__file__).resolve().parent.parent / "virtual memory.py"
    spec = importlib.util.spec_from_file_location("virtual_memory_app", vm_path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载虚拟存储器模块")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "VMApp")


class AppLauncher:
    """Manage one simulation window and avoid duplicates."""

    def __init__(self, master, title, builder):
        self.master = master
        self.title = title
        self.builder = builder
        self.window = None

    def open(self):
        if self.window and self.window.winfo_exists():
            try:
                self.window.lift()
                self.window.focus_force()
            except Exception:
                pass
            return
        self.window = tk.Toplevel(self.master)
        self.window.title(self.title)
        self.window.configure(bg="#f8f9fb")
        self.builder(self.window)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        if self.window:
            try:
                self.window.destroy()
            finally:
                self.window = None


class OSDesktop(tk.Tk):
    """Unified desktop that links all OS visualizations in this repo."""

    def __init__(self):
        super().__init__()
        self.title("OS2 可视化操作系统实验台")
        self.geometry("1080x780")
        self.configure(bg="#eef1f6")
        self._setup_style()

        # Launchers (reuse existing simulators)
        self.launch_mfq = AppLauncher(self, "多级反馈队列调度", lambda root: MFQApp(root))
        self.launch_pager = AppLauncher(self, "基本分页管理", lambda root: PageApp(root))
        self.launch_pv = AppLauncher(self, "生产者-消费者 PV", lambda root: PVApp(root))
        vm_class = _load_virtual_memory_app()
        self.launch_vm = AppLauncher(self, "请求分页虚拟存储器", lambda root: vm_class(root))
        self.launch_pipeline = AppLauncher(self, "作业七态管线", lambda root: KernelSimulator(root))

        self._build_layout()

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#1f3b57", background="#eef1f6")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12), foreground="#4b5b6b", background="#eef1f6")
        style.configure("Section.TLabelframe", background="#eef1f6", relief="flat")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 14, "bold"), foreground="#1f3b57" )
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Card.TLabel", background="#ffffff", font=("Segoe UI", 11), foreground="#3b4b5c")
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))

    def _build_layout(self):
        header = ttk.Frame(self, padding=(18, 14), style="Card.TFrame")
        header.pack(fill="x", padx=18, pady=(18, 10))

        ttk.Label(header, text="OS2 可视化操作系统", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="集中演示调度、存储管理、同步互斥与虚拟内存。点击任意模块即可开启独立可视化窗口。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        actions = ttk.Frame(header, style="Card.TFrame")
        actions.pack(anchor="e", pady=(8, 0))
        ttk.Button(actions, text="全部启动", style="Accent.TButton", command=self._open_all).pack(side="left", padx=6)
        ttk.Button(actions, text="退出", command=self.destroy).pack(side="left", padx=6)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        notebook.add(self._make_overview_tab(notebook), text="概览")
        notebook.add(self._make_scheduler_tab(notebook), text="进程调度")
        notebook.add(self._make_memory_tab(notebook), text="内存管理")
        notebook.add(self._make_virtual_tab(notebook), text="虚拟存储")
        notebook.add(self._make_sync_tab(notebook), text="同步互斥")
        notebook.add(self._make_pipeline_tab(notebook), text="作业流")

    def _make_overview_tab(self, parent):
        tab = ttk.Frame(parent, style="Card.TFrame", padding=18)

        intro = (
            "整合仓库中的所有 Python 实验：\n"
            "• 多级反馈队列：时间片、可抢占、甘特图。\n"
            "• 基本分页：位示图、页表、自动/手动分配。\n"
            "• 虚拟存储：请求分页 + 固定分配置换。\n"
            "• PV 问题：生产者/消费者信号量与缓冲区可视化。"
        )
        ttk.Label(tab, text=intro, style="Subtitle.TLabel", justify="left").pack(anchor="w")

        cards = ttk.Frame(tab, style="Card.TFrame")
        cards.pack(fill="x", pady=(12, 0))

        self._render_card(cards, "多级反馈队列调度", "逐时间片可视化运行，支持动态队列数与导入示例", self.launch_mfq.open)
        self._render_card(cards, "基本分页", "位示图展示页框分配，进程页表同步刷新", self.launch_pager.open)
        self._render_card(cards, "请求分页/置换", "固定分配 FIFO，内存/磁盘双视图，自动生命周期演示", self.launch_vm.open)
        self._render_card(cards, "生产者-消费者", "empty/full/mutex 信号量联动，实时缓冲区动画", self.launch_pv.open)
        self._render_card(cards, "作业七态流转", "统一调度、分页、虚拟存储与 IO 信号量的通信演示", self.launch_pipeline.open)

        return tab

    def _make_scheduler_tab(self, parent):
        tab = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(tab, text="多级反馈队列调度", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            tab,
            text="支持队列数量与时间片动态配置，甘特图实时绘制，进程完成表展示周转时间。",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        self._render_card(tab, "启动调度器", "打开仿真窗口，录入或导入作业，观察抢占/降级流程。", self.launch_mfq.open)
        return tab

    def _make_memory_tab(self, parent):
        tab = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(tab, text="基本分页存储管理", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            tab,
            text="位示图/页表/进程列表组合视图，支持自动步进或手动申请释放页框。",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        self._render_card(tab, "打开分页管理", "设置页框数、创建进程并查看页表映射。", self.launch_pager.open)
        return tab

    def _make_virtual_tab(self, parent):
        tab = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(tab, text="请求分页虚拟存储器", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            tab,
            text="固定分配 FIFO 置换，可视化物理帧与磁盘槽，同步日志追踪缺页与置换。",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        self._render_card(tab, "启动虚拟存储", "导入示例、自动轮转访问，展示驻留页与脏页状态。", self.launch_vm.open)
        return tab

    def _make_pipeline_tab(self, parent):
        tab = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(tab, text="作业全链路（七态 + 模块通信）", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            tab,
            text="作业依次经历新建/就绪/运行/阻塞/挂起/完成，运行时触发分页、缺页置换与 IO 信号量，展示模块间协作。",
            style="Subtitle.TLabel",
            wraplength=820,
            justify="left",
        ).pack(anchor="w")
        self._render_card(tab, "打开作业管线", "一步到位观察七态图与调度-内存-IO 联动。", self.launch_pipeline.open)
        return tab

    def _make_sync_tab(self, parent):
        tab = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(tab, text="生产者-消费者 (PV)", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            tab,
            text="empty/full/mutex 信号量驱动的缓冲区动画，日志实时滚动，展示并发与互斥。",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        self._render_card(tab, "运行 PV", "打开生产者/消费者可视化，观察信号量取放资源。", self.launch_pv.open)
        return tab

    def _render_card(self, parent, title, desc, command):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12))
        card.pack(fill="x", pady=8)
        ttk.Label(card, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text=desc, style="Card.TLabel").pack(anchor="w", pady=(4, 8))
        ttk.Button(card, text="打开", style="Accent.TButton", command=command).pack(anchor="w")

    def _open_all(self):
        try:
            self.launch_mfq.open()
            self.launch_pager.open()
            self.launch_vm.open()
            self.launch_pv.open()
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))


def run():
    app = OSDesktop()
    app.mainloop()


if __name__ == "__main__":
    run()
