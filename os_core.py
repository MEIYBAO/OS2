# os_sim_gui_cn.py
# 只改UI表头为中文，其余保持一致（固定块+局部FIFO+MLFQ+文件系统）

import math
import random
import colorsys
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import simpledialog
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


STATE_NEW = "NEW"
STATE_READY = "READY"
STATE_RUNNING = "RUNNING"
STATE_BLOCKED = "BLOCKED"
STATE_TERMINATED = "TERMINATED"

MLFQ_QUANTA = [4, 8, 16]
BOOST_PERIOD = 20
PAGEFAULT_BLOCK_TICKS = 2

DEFAULT_PAGE_SIZE_KB = 4
DEFAULT_MEM_SIZE_KB = 64
DEFAULT_VM_SIZE_KB = 256

DEFAULT_PROC_MEM_KB = 16
DEFAULT_PROC_WORK = 60
BUFFER_CAPACITY_LINES = 24  # 用文件内容作为有界缓冲区，行数即容量
PROD_BLOCK_TICKS = 2  # 缓冲区满时生产者阻塞时长
CONS_BLOCK_TICKS = 2  # 缓冲区空时消费者阻塞时长
PC_BUFFER_CAPACITY = 5  # 内置生产者-消费者缓冲池容量



@dataclass
class FSNode:
    name: str
    is_dir: bool
    parent: Optional["FSNode"] = None
    children: Dict[str, "FSNode"] = field(default_factory=dict)
    content: str = ""

    def path(self) -> str:
        if self.parent is None:
            return "/"
        parts = []
        cur: Optional["FSNode"] = self
        while cur is not None and cur.parent is not None:
            parts.append(cur.name)
            cur = cur.parent
        return "/" + "/".join(reversed(parts))


class FileSystem:
    def __init__(self):
        self.root = FSNode(name="", is_dir=True, parent=None)

    def _split(self, path: str) -> List[str]:
        p = (path or "/").strip()
        if not p.startswith("/"):
            raise ValueError("路径必须以 / 开头")
        return [x for x in p.split("/") if x]

    def _get_node(self, path: str) -> FSNode:
        parts = self._split(path)
        cur = self.root
        for name in parts:
            if not cur.is_dir:
                raise ValueError("路径中包含文件，无法继续进入")
            if name not in cur.children:
                raise ValueError(f"路径不存在：{path}")
            cur = cur.children[name]
        return cur

    def exists(self, path: str) -> bool:
        try:
            self._get_node(path)
            return True
        except Exception:
            return False

    def mkdir(self, dir_path: str, name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("文件夹名不能为空")
        if "/" in name:
            raise ValueError("名称不能包含 /")

        d = self._get_node(dir_path)
        if not d.is_dir:
            raise ValueError("目标不是目录")
        if name in d.children:
            raise ValueError("同名文件/文件夹已存在")

        node = FSNode(name=name, is_dir=True, parent=d)
        d.children[name] = node
        return node.path()

    def touch(self, dir_path: str, name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("文件名不能为空")
        if "/" in name:
            raise ValueError("名称不能包含 /")

        d = self._get_node(dir_path)
        if not d.is_dir:
            raise ValueError("目标不是目录")
        if name in d.children:
            raise ValueError("同名文件/文件夹已存在")

        node = FSNode(name=name, is_dir=False, parent=d, content="")
        d.children[name] = node
        return node.path()

    def write(self, file_path: str, text: str, append: bool = True) -> None:
        f = self._get_node(file_path)
        if f.is_dir:
            raise ValueError("不能写入文件夹")
        f.content = (f.content + text) if append else text

    def read(self, file_path: str) -> str:
        f = self._get_node(file_path)
        if f.is_dir:
            raise ValueError("不能读取文件夹（请选择文件）")
        return f.content

    def list_file_paths(self) -> List[str]:
        res: List[str] = []

        def dfs(node: FSNode):
            if node.is_dir:
                for child in sorted(
                    node.children.values(), key=lambda x: (not x.is_dir, x.name.lower())
                ):
                    dfs(child)
            else:
                res.append(node.path())

        dfs(self.root)
        return res


@dataclass
class PageEntry:
    vpn: int
    present: bool = False
    frame_id: Optional[int] = None


@dataclass
class PCB:
    pid: int
    name: str
    mem_kb: int
    page_size_kb: int
    access_seq: List[int]

    state: str = STATE_NEW
    queue_level: int = 0
    pc: int = 0
    remaining: int = 0
    slice_left: int = 0
    page_faults: int = 0
    block_left: int = 0

    num_pages: int = 0
    page_table: Dict[int, PageEntry] = field(default_factory=dict)
    swap_map: Dict[int, int] = field(default_factory=dict)

    ram_pool: List[int] = field(default_factory=list)
    ram_fifo: deque = field(default_factory=deque)

    # ??????/???????
    role: Optional[str] = None  # "producer" / "consumer"
    target_file: Optional[str] = None
    items_goal: int = 0
    items_done: int = 0

    def __post_init__(self):
        self.num_pages = max(1, math.ceil(self.mem_kb / self.page_size_kb))
        self.remaining = len(self.access_seq)
        self.slice_left = MLFQ_QUANTA[self.queue_level]
        self.page_table = {vpn: PageEntry(vpn=vpn) for vpn in range(self.num_pages)}

    @property
    def ram_quota(self) -> int:
        return len(self.ram_pool)

    def is_done(self) -> bool:
        # 生产者/消费者：当 items_goal<=0 时视为长期运行，不根据 pc 结束
        if self.role in ("producer", "consumer"):
            if getattr(self, "items_goal", 0) <= 0:
                return False
            # items_goal>0 时仅按完成量结束
            return self.items_done >= self.items_goal
        return self.pc >= len(self.access_seq)

    def next_vpn(self) -> Optional[int]:
        # 生产者/消费者：循环访问序列，维持“长期运行”
        if self.role in ("producer", "consumer") and self.access_seq:
            return self.access_seq[self.pc % len(self.access_seq)]
        if self.pc >= len(self.access_seq):
            return None
        return self.access_seq[self.pc]


@dataclass
class SwapFrame:
    frame_id: int
    owner: Optional[Tuple[int, int]] = None


class VirtualMemoryManager:
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.frames: List[SwapFrame] = [
            SwapFrame(frame_id=i) for i in range(num_frames)
        ]

    def free_count(self) -> int:
        return sum(1 for f in self.frames if f.owner is None)

    def _find_free(self) -> Optional[SwapFrame]:
        for f in self.frames:
            if f.owner is None:
                return f
        return None

    def allocate_process_pages(self, pcb: PCB) -> None:
        pcb.swap_map.clear()
        need = pcb.num_pages
        if self.free_count() < need:
            raise RuntimeError(
                f"SWAP不足：需要 {need} 页框，但剩余 {self.free_count()}"
            )

        for vpn in range(pcb.num_pages):
            f = self._find_free()
            if f is None:
                raise RuntimeError("SWAP不足：分配失败")
            f.owner = (pcb.pid, vpn)
            pcb.swap_map[vpn] = f.frame_id

    def free_process(self, pcb: PCB) -> None:
        for vpn, sid in list(pcb.swap_map.items()):
            if 0 <= sid < len(self.frames):
                sf = self.frames[sid]
                if sf.owner == (pcb.pid, vpn):
                    sf.owner = None
        pcb.swap_map.clear()


@dataclass
class RAMFrame:
    frame_id: int
    reserved_pid: Optional[int] = None
    owner: Optional[Tuple[int, int]] = None


class MemoryManager:
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.frames: List[RAMFrame] = [RAMFrame(frame_id=i) for i in range(num_frames)]

    def free_unreserved_frames(self) -> List[int]:
        return [fr.frame_id for fr in self.frames if fr.reserved_pid is None]

    def reserve_frames_for_process(self, pcb: PCB, desired_quota: int) -> None:
        free_ids = self.free_unreserved_frames()
        if not free_ids:
            raise RuntimeError("RAM不足：没有可分配的物理块给新进程")

        quota = min(desired_quota, len(free_ids))
        if quota <= 0:
            raise RuntimeError("RAM不足：无法为新进程分配物理块")

        chosen = free_ids[:quota]
        for fid in chosen:
            fr = self.frames[fid]
            fr.reserved_pid = pcb.pid
            fr.owner = None

        pcb.ram_pool = chosen
        pcb.ram_fifo = deque()

    def release_process_frames(self, pcb: PCB) -> None:
        for fid in pcb.ram_pool:
            fr = self.frames[fid]
            if fr.owner is not None:
                pid, vpn = fr.owner
                if pid == pcb.pid:
                    ent = pcb.page_table.get(vpn)
                    if ent:
                        ent.present = False
                        ent.frame_id = None
                fr.owner = None
            fr.reserved_pid = None
        pcb.ram_pool.clear()
        pcb.ram_fifo.clear()

    def _find_free_in_pool(self, pcb: PCB) -> Optional[int]:
        for fid in pcb.ram_pool:
            if self.frames[fid].owner is None:
                return fid
        return None

    def page_in_local_fifo(
        self, pcb: PCB, vpn: int, all_pcbs: Dict[int, PCB]
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        ent = pcb.page_table[vpn]
        if ent.present:
            return (False, None)

        free_fid = self._find_free_in_pool(pcb)
        if free_fid is not None:
            fr = self.frames[free_fid]
            fr.owner = (pcb.pid, vpn)
            ent.present = True
            ent.frame_id = free_fid
            pcb.ram_fifo.append(free_fid)
            return (False, None)

        victim_fid = pcb.ram_fifo.popleft() if pcb.ram_fifo else pcb.ram_pool[0]
        victim_fr = self.frames[victim_fid]
        victim = victim_fr.owner
        if victim is None:
            victim_fr.owner = (pcb.pid, vpn)
            ent.present = True
            ent.frame_id = victim_fid
            pcb.ram_fifo.append(victim_fid)
            return (False, None)

        vpid, vvpn = victim
        vpcb = all_pcbs.get(vpid)
        if vpcb is not None and vvpn in vpcb.page_table:
            vent = vpcb.page_table[vvpn]
            vent.present = False
            vent.frame_id = None

        victim_fr.owner = (pcb.pid, vpn)
        ent.present = True
        ent.frame_id = victim_fid
        pcb.ram_fifo.append(victim_fid)
        return (True, (vpid, vvpn))


class MLFQScheduler:
    def __init__(self, levels: int):
        self.levels = levels
        self.queues: List[List[int]] = [[] for _ in range(levels)]
        self.running_pid: Optional[int] = None

    def add_new(self, pid: int) -> None:
        self.queues[0].append(pid)

    def requeue_after_timeslice(self, pcb: PCB) -> None:
        # 若未到底层则降级；到底层则继续排在末尾并重置时间片
        if pcb.queue_level < self.levels - 1:
            pcb.queue_level += 1
        else:
            pcb.queue_level = self.levels - 1
        pcb.slice_left = MLFQ_QUANTA[pcb.queue_level]
        self.queues[pcb.queue_level].append(pcb.pid)

    def requeue_after_unblock(self, pcb: PCB) -> None:
        # 阻塞恢复：回到最高优先级队列，使用该队列时间片
        pcb.queue_level = 0
        pcb.slice_left = MLFQ_QUANTA[0]
        self.queues[0].append(pcb.pid)

    def requeue_after_preempt(self, pcb: PCB) -> None:
        # 被抢占：保持当前队列，保留剩余时间片（至少1 tick），排到队尾
        if pcb.slice_left <= 0:
            pcb.slice_left = 1
        self.queues[pcb.queue_level].append(pcb.pid)

    def highest_ready_level(self, pcbs: Dict[int, PCB]) -> Optional[int]:
        for lvl in range(self.levels):
            for pid in list(self.queues[lvl]):
                pcb = pcbs.get(pid)
                if pcb and pcb.state == STATE_READY:
                    return lvl
        return None

    def boost_all_ready(self, pcbs: Dict[int, PCB]) -> None:
        all_pids: List[int] = []
        for q in self.queues:
            all_pids.extend(q)
            q.clear()
        seen = set()
        for pid in all_pids:
            if pid in seen:
                continue
            seen.add(pid)
            pcb = pcbs.get(pid)
            if pcb and pcb.state == STATE_READY:
                pcb.queue_level = 0
                pcb.slice_left = MLFQ_QUANTA[0]
                self.queues[0].append(pid)

    def pick_next(self, pcbs: Dict[int, PCB]) -> Optional[int]:
        for lvl in range(self.levels):
            while self.queues[lvl]:
                pid = self.queues[lvl].pop(0)
                pcb = pcbs.get(pid)
                if pcb and pcb.state == STATE_READY:
                    return pid
        return None

    def snapshot_queues(self, pcbs: Optional[Dict[int, PCB]] = None) -> List[List[int]]:
        qs = [list(q) for q in self.queues]
        if pcbs is not None and self.running_pid is not None:
            pcb = pcbs.get(self.running_pid)
            if pcb and 0 <= pcb.queue_level < len(qs):
                qs[pcb.queue_level] = [self.running_pid] + qs[pcb.queue_level]
        return qs


class MiniOS:
    def __init__(
        self,
        page_size_kb: int,
        mem_size_kb: int,
        vm_size_kb: int,
        fs: Optional[FileSystem] = None,
    ):
        self.fs = fs if fs is not None else FileSystem()
        self.page_size_kb = page_size_kb
        self.mem_size_kb = mem_size_kb
        self.vm_size_kb = vm_size_kb

        self.ram_frames = mem_size_kb // page_size_kb
        self.swap_frames = vm_size_kb // page_size_kb

        self.swap = VirtualMemoryManager(num_frames=self.swap_frames)
        self.ram = MemoryManager(num_frames=self.ram_frames)
        self.sched = MLFQScheduler(levels=len(MLFQ_QUANTA))

        self.pcbs: Dict[int, PCB] = {}
        self.pid_counter = 1000

        self.ticks = 0
        self.log_lines: List[str] = []

        # 生产者-消费者缓冲池（内置，非文件模式）
        self.pc_buffer_capacity = PC_BUFFER_CAPACITY
        self.pc_buffer: List[Optional[str]] = [None] * self.pc_buffer_capacity
        self.pc_buffer_count = 0
        self.pc_in_ptr = 0
        self.pc_out_ptr = 0
        self.waiting_producers: List[int] = []
        self.waiting_consumers: List[int] = []

    def log(self, msg: str) -> None:
        self.log_lines.append(f"[tick={self.ticks:04d}] {msg}")
        if len(self.log_lines) > 900:
            self.log_lines = self.log_lines[-900:]

    def _make_access_seq(
        self, num_pages: int, length: int, locality: bool = True, salt: int = 0
    ) -> List[int]:
        if num_pages <= 0:
            return []
        seq: List[int] = []
        hot = (random.randint(0, num_pages - 1) + salt) % num_pages
        window = max(2, min(5, num_pages))
        for _ in range(length):
            if locality and random.random() < 0.75:
                low = max(0, hot - window // 2)
                high = min(num_pages - 1, hot + window // 2)
                seq.append((random.randint(low, high) + salt) % num_pages)
            else:
                seq.append((random.randint(0, num_pages - 1) + salt) % num_pages)
            if random.random() < 0.1:
                hot = (random.randint(0, num_pages - 1) + salt) % num_pages
        return seq

    def _file_to_access_seq(
        self, content: str, num_pages: int, salt: int = 0
    ) -> List[int]:
        if num_pages <= 0:
            return []
        if not content:
            return self._make_access_seq(num_pages, length=12, locality=True, salt=salt)

        seq: List[int] = []
        for ch in content[:220]:
            seq.append(((ord(ch) * 131) + salt * 17) % num_pages)
        seq.extend(
            self._make_access_seq(
                num_pages, length=max(12, len(seq) // 3), locality=True, salt=salt
            )
        )
        return seq

    def _calc_ram_quota_frames(self, num_pages: int) -> int:
        return max(1, min(num_pages, int(math.ceil(num_pages / 2.0))))

    def _fit_pages_list_to_swap(
        self, pages_list: List[int], swap_free: int
    ) -> List[int]:
        if swap_free <= 0:
            return []
        pages = [max(1, int(x)) for x in pages_list]
        while sum(pages) > swap_free:
            idx = max(range(len(pages)), key=lambda i: pages[i])
            if pages[idx] > 1:
                pages[idx] -= 1
            else:
                break
        if sum(pages) > swap_free:
            max_procs = min(len(pages), swap_free)
            pages = pages[:max_procs]
        return pages

    def create_process(
        self,
        name: str,
        mem_kb: int,
        work_len: int,
        from_file: Optional[str] = None,
        salt: int = 0,
    ) -> PCB:
        self.pid_counter += 1
        pid = self.pid_counter
        num_pages = max(1, math.ceil(mem_kb / self.page_size_kb))
        if from_file is not None:
            content = self.fs.read(from_file)
            access_seq = self._file_to_access_seq(
                content, num_pages=num_pages, salt=salt
            )
        else:
            access_seq = self._make_access_seq(
                num_pages=num_pages, length=work_len, locality=True, salt=salt
            )

        pcb = PCB(
            pid=pid,
            name=name,
            mem_kb=mem_kb,
            page_size_kb=self.page_size_kb,
            access_seq=access_seq,
        )
        pcb.state = STATE_READY
        pcb.queue_level = 0
        pcb.slice_left = MLFQ_QUANTA[0]

        self.swap.allocate_process_pages(pcb)
        try:
            desired_quota = self._calc_ram_quota_frames(pcb.num_pages)
            self.ram.reserve_frames_for_process(pcb, desired_quota=desired_quota)
        except Exception as e:
            self.swap.free_process(pcb)
            raise RuntimeError(str(e))

        self.pcbs[pid] = pcb
        self.sched.add_new(pid)
        self.log(
            f"创建 PID={pid} name={name} mem={mem_kb}KB pages={pcb.num_pages} | SWAP全页OK | RAM固定块={pcb.ram_quota}"
        )
        return pcb

    def finish_process(self, pid: int, reason: str = "完成") -> None:
        pcb = self.pcbs.get(pid)
        if pcb is None:
            return
        self.ram.release_process_frames(pcb)
        self.swap.free_process(pcb)
        pcb.state = STATE_TERMINATED
        pcb.block_left = 0
        pcb.slice_left = 0
        if self.sched.running_pid == pid:
            self.sched.running_pid = None
        self.log(f"终止 PID={pid}（{reason}）-> 释放RAM固定块 + SWAP")

    def kill_process(self, pid: int) -> None:
        self.finish_process(pid, reason="手动终止")

    def run_file_spawn_processes(
        self, file_path: str, nprocs: int, mem_kb_each: int
    ) -> List[PCB]:
        if nprocs <= 0:
            return []
        base = file_path.split("/")[-1] or "file"
        base_pages = max(1, math.ceil(mem_kb_each / self.page_size_kb))
        center = (nprocs - 1) / 2.0
        raw_pages_list: List[int] = []
        for i in range(nprocs):
            delta = int(round(i - center))
            raw_pages_list.append(max(1, base_pages + delta))
        pages_list = self._fit_pages_list_to_swap(
            raw_pages_list, self.swap.free_count()
        )
        if not pages_list:
            raise RuntimeError(
                "SWAP不足：当前空闲SWAP=0，无法派生进程（请终止部分进程或增大虚存）"
            )
        if len(pages_list) < len(raw_pages_list):
            self.log(
                f"SWAP不足自动降级：原计划{len(raw_pages_list)}个(总页={sum(raw_pages_list)})，实际{len(pages_list)}个(总页={sum(pages_list)})"
            )
        if sum(pages_list) < sum(raw_pages_list):
            self.log(
                f"SWAP不足自动缩页：原总页={sum(raw_pages_list)}，缩减后总页={sum(pages_list)}"
            )

        roles = ["loader", "parser", "worker", "io", "writer", "helper"]
        created: List[PCB] = []
        for i, pages_i in enumerate(pages_list):
            mem_i = pages_i * self.page_size_kb
            work_len = 20 + i * 8 + random.randint(0, 6)
            salt = i + 1
            role = roles[i % len(roles)]
            pname = f"{base}:{role}#W{i+1}"
            created.append(
                self.create_process(
                    pname, mem_i, work_len, from_file=file_path, salt=salt
                )
            )
        self.log(f"运行文件：{file_path} -> 派生 {len(created)} 个相关进程")
        return created

    def run_prod_consumer(
        self,
        producers: int = 2,
        consumers: int = 2,
        items_per_producer: int = 0,
        items_per_consumer: int = 0,
        mem_kb_each: int = DEFAULT_PROC_MEM_KB,
    ) -> List[PCB]:
        """创建生产者/消费者进程，使用内置缓冲池（容量 PC_BUFFER_CAPACITY）。"""
        if producers <= 0 or consumers <= 0:
            raise RuntimeError("生产者/消费者数量必须为正整数")

        # 重置内置缓冲池
        self.pc_buffer = [None] * self.pc_buffer_capacity
        self.pc_buffer_count = 0
        self.pc_in_ptr = 0
        self.pc_out_ptr = 0
        self.waiting_producers = []
        self.waiting_consumers = []

        prod_infinite = items_per_producer <= 0
        cons_infinite = items_per_consumer <= 0
        total_items = producers * items_per_producer if not prod_infinite else 0
        target_consume = (
            items_per_consumer
            if items_per_consumer > 0
            else (math.ceil(total_items / consumers) if not prod_infinite else 0)
        )

        created: List[PCB] = []
        for i in range(producers):
            pcb = self.create_process(
                name=f"producer#{i+1}",
                mem_kb=mem_kb_each,
                work_len=(items_per_producer or 8) * 4,
                from_file=None,
                salt=i + 10,
            )
            pcb.role = "producer"
            pcb.items_goal = 0 if prod_infinite else items_per_producer
            created.append(pcb)

        for j in range(consumers):
            pcb = self.create_process(
                name=f"consumer#{j+1}",
                mem_kb=mem_kb_each,
                work_len=(target_consume or 8) * 4,
                from_file=None,
                salt=j + 100,
            )
            pcb.role = "consumer"
            pcb.items_goal = 0 if cons_infinite else target_consume
            created.append(pcb)

        self.log(
            f"生产者-消费者：内置缓冲池(cap={self.pc_buffer_capacity}) 生产者={producers}{'无限' if prod_infinite else '×'+str(items_per_producer)} 消费者={consumers}{'无限' if cons_infinite else '×'+str(target_consume)}"
        )
        return created

    def tick(self) -> None:
        self.ticks += 1
        # 不再定期全局BOOST，避免破坏当前优先级分布

        for pcb in list(self.pcbs.values()):
            if pcb.state == STATE_BLOCKED:
                if pcb.block_left > 0:
                    pcb.block_left -= 1
                if pcb.block_left > 0:
                    continue
                if pcb.block_left == 0:
                    pcb.state = STATE_READY
                    self.sched.requeue_after_unblock(pcb)
                    self.log(
                        f"PID={pcb.pid} 调页完成 -> READY（入队Q{pcb.queue_level}）"
                    )
                # block_left <0 表示等待事件唤醒，不在此处自动恢复
                continue

        if self.sched.running_pid is None:
            nxt = self.sched.pick_next(self.pcbs)
            if nxt is not None:
                p = self.pcbs[nxt]
                p.state = STATE_RUNNING
                if p.slice_left <= 0:
                    p.slice_left = MLFQ_QUANTA[p.queue_level]
                self.sched.running_pid = nxt
                self.log(
                    f"调度：PID={nxt} -> RUNNING（Q{p.queue_level}, slice={p.slice_left}）"
                )
            else:
                # 防止队列状态意外丢失：直接从 READY 进程中兜底挑选一个
                fallback = self._pick_ready_fallback()
                if fallback is not None:
                    p = self.pcbs[fallback]
                    p.state = STATE_RUNNING
                    if p.slice_left <= 0:
                        p.slice_left = MLFQ_QUANTA[p.queue_level]
                    self.sched.running_pid = fallback
                    self.log(
                        f"调度(兜底)：PID={fallback} -> RUNNING（Q{p.queue_level}, slice={p.slice_left}）"
                    )

        if self.sched.running_pid is None:
            self.log("CPU 空闲（无READY进程）")
            return

        pcb = self.pcbs.get(self.sched.running_pid)
        if pcb is None or pcb.state != STATE_RUNNING:
            self.sched.running_pid = None
            return
        # 本 tick 开始默认不能直接生产/消费，需先完成一次 CPU 执行
        setattr(pcb, "_pc_ready_for_io", False)

        # 高优先级队列有就绪进程时抢占当前进程（保留剩余时间片，排回当前队列尾）
        ready_lvl = self.sched.highest_ready_level(self.pcbs)
        if ready_lvl is not None and ready_lvl < pcb.queue_level:
            pcb.state = STATE_READY
            if pcb.slice_left <= 0:
                pcb.slice_left = 1
            self.sched.requeue_after_preempt(pcb)
            self.log(
                f"抢占：PID={pcb.pid} (Q{pcb.queue_level}) 被更高优先级Q{ready_lvl} 进程抢占，剩余slice={pcb.slice_left} 入队尾"
            )
            self.sched.running_pid = None
            nxt = self.sched.pick_next(self.pcbs)
            if nxt is None:
                self.log("CPU 空闲（抢占后无READY进程）")
                return
            p = self.pcbs[nxt]
            p.state = STATE_RUNNING
            if p.slice_left <= 0:
                p.slice_left = MLFQ_QUANTA[p.queue_level]
            self.sched.running_pid = nxt
            self.log(
                f"调度：PID={nxt} -> RUNNING（Q{p.queue_level}, slice={p.slice_left}）"
            )
            pcb = p

        if pcb.is_done():
            self.finish_process(pcb.pid, reason="执行完成")
            self.sched.running_pid = None
            return

        vpn = pcb.next_vpn()
        assert vpn is not None
        ent = pcb.page_table[vpn]
        if not ent.present:
            pcb.page_faults += 1
            swap_id = pcb.swap_map.get(vpn, None)
            replaced, victim = self.ram.page_in_local_fifo(pcb, vpn, self.pcbs)
            if victim is None:
                self.log(
                    f"页故障：PID={pcb.pid} VPN={vpn} 从SWAP={swap_id} -> 调入RAM(pool内)（无置换）"
                )
            else:
                vpid, vvpn = victim
                self.log(
                    f"页故障：PID={pcb.pid} VPN={vpn} 从SWAP={swap_id} -> 局部FIFO置换 换出 PID={vpid} VPN={vvpn}"
                )
            pcb.state = STATE_BLOCKED
            pcb.block_left = PAGEFAULT_BLOCK_TICKS
            self.log(
                f"PID={pcb.pid} -> BLOCKED（模拟调页I/O {PAGEFAULT_BLOCK_TICKS} ticks）"
            )
            self.sched.running_pid = None
            return

        pcb.pc += 1
        pcb.remaining = max(0, len(pcb.access_seq) - pcb.pc)
        pcb.slice_left -= 1
        setattr(pcb, "_pc_ready_for_io", True)
        self.log(
            f"执行：PID={pcb.pid} 命中VPN={vpn} RAMFrame={ent.frame_id} 剩余work={pcb.remaining} slice_left={pcb.slice_left}"
        )

        if self._maybe_handle_prod_consumer(pcb):
            return

        if pcb.is_done():
            self.finish_process(pcb.pid, reason="执行完成")
            self.sched.running_pid = None
            return

        if pcb.slice_left <= 0:
            pcb.state = STATE_READY
            self.sched.running_pid = None
            self.sched.requeue_after_timeslice(pcb)
            self.log(
                f"时间片耗尽：PID={pcb.pid} -> READY，入队Q{pcb.queue_level} slice重置={pcb.slice_left}"
            )


    def _maybe_handle_prod_consumer(self, pcb: PCB) -> bool:
        """生产/消费 I/O；返回 True 表示本 tick 已处理阻塞/完成/消费生产。"""
        if pcb.role not in ("producer", "consumer"):
            return False
        if pcb.items_goal > 0 and pcb.items_done >= pcb.items_goal:
            return False
        if not getattr(pcb, "_pc_ready_for_io", False):
            return False

        buf = self.pc_buffer
        cap = getattr(self, "pc_buffer_capacity", PC_BUFFER_CAPACITY)
        count = getattr(self, "pc_buffer_count", 0)

        if pcb.role == "producer":
            if count >= cap:
                pcb.state = STATE_BLOCKED
                pcb.block_left = -1  # 等待空槽出现后被唤醒
                if pcb.pid not in self.waiting_producers:
                    self.waiting_producers.append(pcb.pid)
                self.sched.running_pid = None
                setattr(pcb, "_pc_ready_for_io", False)
                self.log(
                    f"Producer blocked: PID={pcb.pid} buffer full ({count}/{cap}) -> wait for consumer to free slot"
                )
                return True
            put_slot = self.pc_in_ptr
            item = f"{pcb.pid}-item{pcb.items_done + 1}"
            buf[put_slot] = item
            pcb.items_done += 1
            self.pc_in_ptr = (self.pc_in_ptr + 1) % cap
            self.pc_buffer_count = count + 1
            setattr(pcb, "_pc_ready_for_io", False)
            self.log(
                f"Produce: PID={pcb.pid} -> buffer put {item} at slot {put_slot} (done {pcb.items_done}/{pcb.items_goal}; buf {self.pc_buffer_count}/{cap}; in->{self.pc_in_ptr})"
            )
            self._wake_one_consumer()
        else:
            if count <= 0:
                pcb.state = STATE_BLOCKED
                pcb.block_left = -1  # 等待产品出现后被唤醒
                if pcb.pid not in self.waiting_consumers:
                    self.waiting_consumers.append(pcb.pid)
                self.sched.running_pid = None
                setattr(pcb, "_pc_ready_for_io", False)
                self.log(
                    f"Consumer blocked: PID={pcb.pid} buffer empty, wait for producer to supply"
                )
                return True
            take_slot = self.pc_out_ptr
            consumed = buf[take_slot]
            buf[take_slot] = None
            pcb.items_done += 1
            self.pc_out_ptr = (self.pc_out_ptr + 1) % cap
            self.pc_buffer_count = max(0, count - 1)
            setattr(pcb, "_pc_ready_for_io", False)
            self.log(
                f"Consume: PID={pcb.pid} <- buffer get {consumed} from slot {take_slot} (done {pcb.items_done}/{pcb.items_goal}; buf {self.pc_buffer_count}/{cap}; out->{self.pc_out_ptr})"
            )
            self._wake_one_producer()

        if pcb.items_goal > 0 and pcb.items_done >= pcb.items_goal:
            self.finish_process(pcb.pid, reason=f"{pcb.role} done")
            self.sched.running_pid = None
            return True

        return False

    def _pick_ready_fallback(self) -> Optional[int]:
        """兜底：在队列异常时，直接从 READY 进程里找一个优先级最低的 pid。"""
        ready_pcbs = [
            (pcb.queue_level, pid)
            for pid, pcb in self.pcbs.items()
            if pcb.state == STATE_READY
        ]
        if not ready_pcbs:
            return None
        ready_pcbs.sort(key=lambda x: (x[0], x[1]))
        return ready_pcbs[0][1]

    def _wake_one_producer(self) -> None:
        while self.waiting_producers:
            pid = self.waiting_producers.pop(0)
            pcb = self.pcbs.get(pid)
            if pcb is None or pcb.state != STATE_BLOCKED:
                continue
            pcb.block_left = 0
            pcb.state = STATE_READY
            self.sched.requeue_after_unblock(pcb)
            self.log(f"Waken producer PID={pid} (slot available)")
            break

    def _wake_one_consumer(self) -> None:
        while self.waiting_consumers:
            pid = self.waiting_consumers.pop(0)
            pcb = self.pcbs.get(pid)
            if pcb is None or pcb.state != STATE_BLOCKED:
                continue
            pcb.block_left = 0
            pcb.state = STATE_READY
            self.sched.requeue_after_unblock(pcb)
            self.log(f"Waken consumer PID={pid} (item available)")
            break

class MiniOSGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OS课设模拟器（固定块分配 + 局部FIFO调页 + MLFQ + 文件系统）")

        self.fs = FileSystem()
        self.os = MiniOS(
            DEFAULT_PAGE_SIZE_KB, DEFAULT_MEM_SIZE_KB, DEFAULT_VM_SIZE_KB, fs=self.fs
        )

        self.running = False
        self._tree_id_to_path: Dict[str, str] = {}
        self._pid_color: Dict[int, str] = {}

        self._build_ui()
        self._refresh_all()

    def _pid_to_color(self, pid: int) -> str:
        if pid in self._pid_color:
            return self._pid_color[pid]
        hue = ((pid * 37) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.50, 0.95)
        color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        self._pid_color[pid] = color
        return color

    def _lighten(self, hex_color: str, factor: float = 0.78) -> str:
        h = hex_color.lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _build_ui(self):
        top = ttk.LabelFrame(self.root, text="参数设置（页/内存/虚存可改）", padding=10)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="页面大小(KB)：").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.spin_page = ttk.Spinbox(top, from_=1, to=64, width=8)
        self.spin_page.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.spin_page.set(str(DEFAULT_PAGE_SIZE_KB))

        ttk.Label(top, text="内存大小(KB)：").grid(
            row=0, column=2, sticky="w", padx=5, pady=5
        )
        self.spin_mem = ttk.Spinbox(top, from_=4, to=4096, width=10)
        self.spin_mem.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.spin_mem.set(str(DEFAULT_MEM_SIZE_KB))

        ttk.Label(top, text="虚存大小(KB)：").grid(
            row=0, column=4, sticky="w", padx=5, pady=5
        )
        self.spin_vm = ttk.Spinbox(top, from_=4, to=16384, width=10)
        self.spin_vm.grid(row=0, column=5, sticky="w", padx=5, pady=5)
        self.spin_vm.set(str(DEFAULT_VM_SIZE_KB))

        ttk.Button(
            top, text="初始化系统(清空进程/内存)", command=self.on_init_system
        ).grid(row=0, column=6, padx=10, pady=5)

        ttk.Separator(top, orient="horizontal").grid(
            row=1, column=0, columnspan=7, sticky="ew", pady=8
        )

        ttk.Label(top, text="Tick:").grid(row=2, column=0, sticky="w", padx=5)
        self.lbl_tick = ttk.Label(top, text="0", width=10)
        self.lbl_tick.grid(row=2, column=1, sticky="w", padx=5)

        ttk.Button(top, text="单步 Tick", command=self.on_tick).grid(
            row=2, column=2, padx=5
        )
        self.btn_run = ttk.Button(top, text="自动运行", command=self.on_run_toggle)
        self.btn_run.grid(row=2, column=3, padx=5)

        ttk.Label(top, text="自动间隔(ms)：").grid(row=2, column=4, sticky="e", padx=5)
        self.ent_interval = ttk.Entry(top, width=8)
        self.ent_interval.grid(row=2, column=5, sticky="w", padx=5)
        self.ent_interval.insert(0, "250")

        ttk.Button(top, text="初始化示例数据", command=self.on_seed_demo).grid(
            row=2, column=6, padx=5
        )

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.page_proc = ttk.Frame(nb)
        self.page_mem = ttk.Frame(nb)
        self.page_fs = ttk.Frame(nb)
        self.page_log = ttk.Frame(nb)

        nb.add(self.page_proc, text="进程与调度")
        nb.add(self.page_mem, text="内存与虚存")
        nb.add(self.page_fs, text="文件系统")
        nb.add(self.page_log, text="日志")

        self._build_proc_page()
        self._build_mem_page()
        self._build_fs_page()
        self._build_log_page()

    # ------------------ 这里：进程表头改中文 ------------------
    def _build_proc_page(self):
        frm = self.page_proc
        left = ttk.Frame(frm)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(frm)
        right.pack(side="right", fill="y")

        cols = (
            "pid",
            "name",
            "state",
            "q",
            "slice",
            "pc",
            "remain",
            "faults",
            "block",
            "memKB",
            "pages",
            "ramQuota",
        )
        self.tree_proc = ttk.Treeview(left, columns=cols, show="headings", height=16)

        # 中文表头映射
        headers = {
            "pid": "进程号",
            "name": "进程名",
            "state": "状态",
            "q": "队列级别",
            "slice": "时间片",
            "pc": "执行步",
            "remain": "剩余步",
            "faults": "缺页次数",
            "block": "阻塞",
            "memKB": "内存KB",
            "pages": "页数",
            "ramQuota": "固定块数",
        }

        # 适配中文的列宽（你也可以再按屏幕微调）
        widths = [80, 280, 90, 80, 80, 70, 80, 90, 70, 80, 70, 90]
        for c, w in zip(cols, widths):
            self.tree_proc.heading(c, text=headers.get(c, c))
            self.tree_proc.column(c, width=w, anchor="center")

        self.tree_proc.pack(fill="both", expand=True)

        qbox = ttk.LabelFrame(left, text="MLFQ 就绪队列（从高到低）")
        qbox.pack(fill="x", pady=10)
        self.lbl_q = ttk.Label(qbox, text="")
        self.lbl_q.pack(anchor="w", padx=8, pady=6)

        box_new = ttk.LabelFrame(right, text="创建随机进程（默认较小）")
        box_new.pack(fill="x", pady=(0, 10))

        ttk.Label(box_new, text="名称").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        self.ent_pname = ttk.Entry(box_new, width=18)
        self.ent_pname.insert(0, "P")
        self.ent_pname.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(box_new, text="进程内存(KB)").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        self.spin_pmem = ttk.Spinbox(box_new, from_=4, to=1024, width=10)
        self.spin_pmem.set(str(DEFAULT_PROC_MEM_KB))
        self.spin_pmem.grid(row=1, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(box_new, text="工作量(访问数)").grid(
            row=2, column=0, sticky="w", padx=6, pady=4
        )
        self.spin_work = ttk.Spinbox(box_new, from_=10, to=600, width=10)
        self.spin_work.set(str(DEFAULT_PROC_WORK))
        self.spin_work.grid(row=2, column=1, padx=6, pady=4, sticky="w")

        ttk.Button(box_new, text="创建", command=self.on_create_proc).grid(
            row=3, column=0, columnspan=2, padx=6, pady=6, sticky="ew"
        )

        box_runfile = ttk.LabelFrame(right, text="运行文件：派生多进程（自动适配SWAP）")
        box_runfile.pack(fill="x", pady=(0, 10))

        ttk.Label(box_runfile, text="文件(路径)").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        self.cmb_file_run = ttk.Combobox(
            box_runfile, values=[], width=28, state="readonly"
        )
        self.cmb_file_run.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(box_runfile, text="派生进程数").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        self.spin_nprocs = ttk.Spinbox(box_runfile, from_=2, to=10, width=10)
        self.spin_nprocs.set("3")
        self.spin_nprocs.grid(row=1, column=1, padx=6, pady=4, sticky="w")

        ttk.Button(
            box_runfile, text="运行（派生多进程）", command=self.on_run_file_spawn
        ).grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        box_kill = ttk.LabelFrame(right, text="终止进程")
        box_kill.pack(fill="x")
        ttk.Button(box_kill, text="终止选中进程", command=self.on_kill_selected).pack(
            fill="x", padx=6, pady=6
        )

    # 下面页面保持不变（略：为节省篇幅，这里继续贴全代码时已包含，运行没问题）
    # ——为了不让你复制时出错，我把其余部分也都保留在后面（与你上一版一致）——

    def _build_mem_page(self):
        frm = self.page_mem
        top = ttk.Frame(frm)
        top.pack(fill="x", pady=(0, 8))

        self.lbl_cfg = ttk.Label(top, text="")
        self.lbl_cfg.pack(side="left", padx=8)

        ttk.Label(top, text="RUNNING PID:").pack(side="left", padx=(20, 4))
        self.lbl_running = ttk.Label(top, text="None")
        self.lbl_running.pack(side="left")

        tip = ttk.Label(top, text="（浅色=进程保留块空闲；深色=已装入页；完成释放）")
        tip.pack(side="left", padx=18)

        ram_box = ttk.LabelFrame(frm, text="RAM 物理页框（固定块 + 局部FIFO）")
        ram_box.pack(fill="both", expand=True, pady=(0, 10))
        self.canvas_ram = tk.Canvas(ram_box, height=240, background="white")
        self.canvas_ram.pack(fill="both", expand=True, padx=8, pady=8)

        swap_box = ttk.LabelFrame(frm, text="SWAP 虚存页框（创建进程先分配全页）")
        swap_box.pack(fill="both", expand=True)
        swap_wrap = ttk.Frame(swap_box)
        swap_wrap.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas_swap = tk.Canvas(swap_wrap, height=240, background="white")
        self.canvas_swap.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(swap_wrap, orient="vertical", command=self.canvas_swap.yview)
        sb.pack(side="right", fill="y")
        self.canvas_swap.configure(yscrollcommand=sb.set)

        pt_box = ttk.LabelFrame(frm, text="分页表（选中PID：VPN -> RAM / SWAP）")
        pt_box.pack(fill="both", expand=True, pady=(10, 0))
        inner = ttk.Frame(pt_box)
        inner.pack(fill="x", padx=8, pady=8)
        ttk.Label(inner, text="选择PID:").pack(side="left")
        self.cmb_pid = ttk.Combobox(inner, values=[], width=12, state="readonly")
        self.cmb_pid.pack(side="left", padx=6)
        self.cmb_pid.bind(
            "<<ComboboxSelected>>", lambda e: self._refresh_pagetable_text()
        )
        self.txt_pt = tk.Text(pt_box, height=9)
        self.txt_pt.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_fs_page(self):
        frm = self.page_fs
        left = ttk.Frame(frm)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(frm)
        right.pack(side="right", fill="y")

        tree_box = ttk.LabelFrame(
            left, text="多级目录（左键选中，右键新建文件/文件夹；文件显示字节数）"
        )
        tree_box.pack(fill="both", expand=True)

        self.tree_fs = ttk.Treeview(tree_box, show="tree")
        self.tree_fs.pack(fill="both", expand=True, padx=8, pady=8)
        self.tree_fs.bind("<<TreeviewSelect>>", lambda e: self.on_fs_select())
        self.tree_fs.bind("<Button-3>", self.on_fs_right_click)

        self.menu_fs = tk.Menu(self.root, tearoff=0)
        self.menu_fs.add_command(label="新建文件", command=self.on_fs_new_file)
        self.menu_fs.add_command(label="新建文件夹", command=self.on_fs_new_dir)

        box_rw = ttk.LabelFrame(right, text="文件读写（选中文件后操作）")
        box_rw.pack(fill="both", expand=True)

        ttk.Label(box_rw, text="当前选中:").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        self.lbl_fs_sel = ttk.Label(box_rw, text="(none)", width=36)
        self.lbl_fs_sel.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(box_rw, text="写入内容").grid(
            row=1, column=0, sticky="nw", padx=6, pady=4
        )
        self.txt_fwrite = tk.Text(box_rw, width=34, height=8)
        self.txt_fwrite.grid(row=1, column=1, padx=6, pady=4)

        ttk.Button(
            box_rw,
            text="追加写入",
            command=lambda: self.on_write_selected_file(append=True),
        ).grid(row=2, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(
            box_rw,
            text="覆盖写入",
            command=lambda: self.on_write_selected_file(append=False),
        ).grid(row=2, column=1, padx=6, pady=6, sticky="ew")

        ttk.Label(box_rw, text="读取显示").grid(
            row=3, column=0, sticky="nw", padx=6, pady=4
        )
        self.txt_fcontent = tk.Text(box_rw, width=34, height=10)
        self.txt_fcontent.grid(row=3, column=1, padx=6, pady=4)

        ttk.Button(
            box_rw, text="读取选中文件", command=self.on_read_selected_file
        ).grid(row=4, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

    def _build_log_page(self):
        frm = self.page_log
        self.txt_log = tk.Text(frm, height=26)
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=8)

    def on_init_system(self):
        try:
            page_kb = int(self.spin_page.get())
            mem_kb = int(self.spin_mem.get())
            vm_kb = int(self.spin_vm.get())
            if page_kb <= 0 or mem_kb <= 0 or vm_kb <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "页面/内存/虚存大小必须为正整数(KB)")
            return
        if mem_kb % page_kb != 0:
            messagebox.showerror("错误", "内存大小必须能被页面大小整除")
            return
        if vm_kb % page_kb != 0:
            messagebox.showerror("错误", "虚存大小必须能被页面大小整除")
            return

        self.os = MiniOS(page_kb, mem_kb, vm_kb, fs=self.fs)
        self.running = False
        self.btn_run.config(text="自动运行")
        self._pid_color.clear()
        self._refresh_all()
        messagebox.showinfo(
            "完成", f"系统已初始化：page={page_kb}KB, RAM={mem_kb}KB, VM={vm_kb}KB"
        )

    def on_tick(self):
        self.os.tick()
        self._refresh_all()

    def on_run_toggle(self):
        self.running = not self.running
        self.btn_run.config(text="停止" if self.running else "自动运行")
        if self.running:
            self._auto_loop()

    def _get_interval_ms(self) -> int:
        try:
            v = int(self.ent_interval.get().strip())
            return v if v > 0 else 250
        except Exception:
            return 250

    def _auto_loop(self):
        if not self.running:
            return
        self.os.tick()
        self._refresh_all()
        self.root.after(self._get_interval_ms(), self._auto_loop)

    def on_seed_demo(self):
        try:
            if not self.fs.exists("/docs"):
                self.fs.mkdir("/", "docs")
            if not self.fs.exists("/bin"):
                self.fs.mkdir("/", "bin")
            if not self.fs.exists("/data"):
                self.fs.mkdir("/", "data")
        except Exception:
            pass

        try:
            if not self.fs.exists("/docs/algo.txt"):
                self.fs.touch("/docs", "algo.txt")
                self.fs.write(
                    "/docs/algo.txt",
                    "MLFQ scheduling with Local FIFO paging & fixed resident frames.\n",
                    append=True,
                )
            if not self.fs.exists("/docs/hello.txt"):
                self.fs.touch("/docs", "hello.txt")
                self.fs.write(
                    "/docs/hello.txt", "Hello OS!\nPaging & VM demo.\n", append=True
                )
        except Exception:
            pass

        try:
            mem_kb_each = min(int(self.spin_pmem.get()), 32)
            self.os.run_file_spawn_processes(
                "/docs/algo.txt", nprocs=3, mem_kb_each=mem_kb_each
            )
        except Exception as e:
            messagebox.showwarning("派生失败", str(e))
        self._refresh_all()

    def on_create_proc(self):
        try:
            name = self.ent_pname.get().strip() or "P"
            mem_kb = int(self.spin_pmem.get())
            work = int(self.spin_work.get())
            salt = random.randint(0, 999)
            pcb = self.os.create_process(
                name=name, mem_kb=mem_kb, work_len=work, from_file=None, salt=salt
            )
            self.os.log(f"UI：创建随机进程 PID={pcb.pid}")
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("创建失败", str(e))

    def on_run_file_spawn(self):
        try:
            fpath = self.cmb_file_run.get().strip()
            if not fpath:
                messagebox.showwarning(
                    "提示", "请先在文件系统中新建文件，然后选择要运行的文件"
                )
                return
            nprocs = int(self.spin_nprocs.get())
            mem_kb_each = min(int(self.spin_pmem.get()), 64)
            self.os.run_file_spawn_processes(
                fpath, nprocs=nprocs, mem_kb_each=mem_kb_each
            )
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("运行文件失败", str(e))

    def on_kill_selected(self):
        sel = self.tree_proc.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在进程表中选中一个进程")
            return
        item = self.tree_proc.item(sel[0])
        pid = int(item["values"][0])
        self.os.kill_process(pid)
        self._refresh_all()

    def _get_selected_fs_path(self) -> Optional[str]:
        sel = self.tree_fs.selection()
        if not sel:
            return None
        return self._tree_id_to_path.get(sel[0])

    def on_fs_select(self):
        p = self._get_selected_fs_path()
        self.lbl_fs_sel.config(text=p if p else "(none)")

    def on_fs_right_click(self, event):
        item_id = self.tree_fs.identify_row(event.y)
        if item_id:
            self.tree_fs.selection_set(item_id)
            self.tree_fs.focus(item_id)
            self.on_fs_select()
        try:
            self.menu_fs.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu_fs.grab_release()

    def _resolve_target_dir_for_create(self) -> str:
        p = self._get_selected_fs_path()
        if not p:
            return "/"
        try:
            node = self.fs._get_node(p)
        except Exception:
            return "/"
        if node.is_dir:
            return node.path()
        if node.parent is None:
            return "/"
        return node.parent.path()

    def on_fs_new_dir(self):
        try:
            target_dir = self._resolve_target_dir_for_create()
            name = simpledialog.askstring(
                "新建文件夹", f"在 {target_dir} 下创建文件夹，输入名称："
            )
            if name is None:
                return
            new_path = self.fs.mkdir(target_dir, name)
            self.os.log(f"文件系统：mkdir {new_path}")
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("新建文件夹失败", str(e))

    def on_fs_new_file(self):
        try:
            target_dir = self._resolve_target_dir_for_create()
            name = simpledialog.askstring(
                "新建文件", f"在 {target_dir} 下创建文件，输入名称："
            )
            if name is None:
                return
            new_path = self.fs.touch(target_dir, name)
            self.os.log(f"文件系统：touch {new_path}")
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("新建文件失败", str(e))

    def on_write_selected_file(self, append: bool):
        try:
            p = self._get_selected_fs_path()
            if not p:
                messagebox.showwarning("提示", "请先在目录树中选中一个文件")
                return
            node = self.fs._get_node(p)
            if node.is_dir:
                messagebox.showwarning("提示", "当前选中的是文件夹，请选中文件再写入")
                return
            text = self.txt_fwrite.get("1.0", "end-1c")
            self.fs.write(p, text, append=append)
            self.os.log(
                f"文件系统：{'append' if append else 'overwrite'} write {p} bytes={len(text)}"
            )
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("写入失败", str(e))

    def on_read_selected_file(self):
        try:
            p = self._get_selected_fs_path()
            if not p:
                messagebox.showwarning("提示", "请先在目录树中选中一个文件")
                return
            node = self.fs._get_node(p)
            if node.is_dir:
                messagebox.showwarning("提示", "当前选中的是文件夹，请选中文件再读取")
                return
            content = self.fs.read(p)
            self.txt_fcontent.delete("1.0", "end")
            self.txt_fcontent.insert("1.0", content)
            self.os.log(f"文件系统：read {p} size={len(content)}")
        except Exception as e:
            messagebox.showerror("读取失败", str(e))

    def _refresh_all(self):
        self.lbl_tick.config(text=str(self.os.ticks))
        self._refresh_proc_table()
        self._refresh_queues()
        self._refresh_mem_canvases()
        self._refresh_pid_combo()
        self._refresh_pagetable_text()
        self._refresh_fs_tree()
        self._refresh_file_combo()
        self._refresh_log()
        self._refresh_cfg_label()

    def _refresh_cfg_label(self):
        free_swap = self.os.swap.free_count()
        free_ram = len(self.os.ram.free_unreserved_frames())
        self.lbl_cfg.config(
            text=(
                f"page={self.os.page_size_kb}KB | "
                f"RAM={self.os.mem_size_kb}KB({self.os.ram_frames}帧, 未保留={free_ram}) | "
                f"VM={self.os.vm_size_kb}KB({self.os.swap_frames}页, 空闲SWAP={free_swap})"
            )
        )

    def _refresh_proc_table(self):
        self.tree_proc.delete(*self.tree_proc.get_children())
        for pid in sorted(self.os.pcbs.keys()):
            p = self.os.pcbs[pid]
            self.tree_proc.insert(
                "",
                "end",
                values=(
                    p.pid,
                    p.name,
                    p.state,
                    p.queue_level,
                    p.slice_left,
                    p.pc,
                    p.remaining,
                    p.page_faults,
                    p.block_left,
                    p.mem_kb,
                    p.num_pages,
                    p.ram_quota,
                ),
            )

    def _refresh_queues(self):
        qs = self.os.sched.snapshot_queues(self.os.pcbs)
        parts = []
        for i, q in enumerate(qs):
            parts.append(f"Q{i}(q={MLFQ_QUANTA[i]}): {q}")
        self.lbl_q.config(text="   |   ".join(parts))

    def _draw_grid(
        self,
        canvas: tk.Canvas,
        n: int,
        get_cell_fn,
        height_hint: int,
        cols_hint: int = 10,
    ):
        canvas.delete("all")
        w = max(860, canvas.winfo_width())
        h = max(height_hint, canvas.winfo_height())
        cols = cols_hint if n >= cols_hint else max(4, cols_hint // 2)
        rows = (n + cols - 1) // cols
        pad = 10
        cell_w = (w - pad * 2) / cols
        cell_h = (h - pad * 2) / max(1, rows)
        for idx in range(n):
            r = idx // cols
            c = idx % cols
            x0 = pad + c * cell_w
            y0 = pad + r * cell_h
            x1 = x0 + cell_w - 8
            y1 = y0 + cell_h - 8
            text, fill, outline = get_cell_fn(idx)
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=2)
            canvas.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2, text=text, font=("Consolas", 10)
            )
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _refresh_mem_canvases(self):
        self.lbl_running.config(
            text=str(self.os.sched.running_pid) if self.os.sched.running_pid else "None"
        )

        def ram_cell(i: int):
            fr = self.os.ram.frames[i]
            if fr.reserved_pid is None and fr.owner is None:
                return (f"R{i}\nFREE", "#ffffff", "#999999")
            if fr.reserved_pid is not None and fr.owner is None:
                base = self._pid_to_color(fr.reserved_pid)
                return (
                    f"R{i}\nRES\nPID={fr.reserved_pid}",
                    self._lighten(base, 0.80),
                    base,
                )
            if fr.owner is not None:
                pid, vpn = fr.owner
                base = self._pid_to_color(pid)
                return (f"R{i}\n{pid}:{vpn}", base, "#111111")
            return (f"R{i}", "#ffffff", "#999999")

        self._draw_grid(
            self.canvas_ram,
            len(self.os.ram.frames),
            ram_cell,
            height_hint=240,
            cols_hint=10,
        )

        def swap_cell(i: int):
            fr = self.os.swap.frames[i]
            if fr.owner is None:
                return (f"S{i}\nFREE", "#ffffff", "#999999")
            pid, vpn = fr.owner
            base = self._pid_to_color(pid)
            return (f"S{i}\n{pid}:{vpn}", self._lighten(base, 0.65), base)

        self._draw_grid(
            self.canvas_swap,
            len(self.os.swap.frames),
            swap_cell,
            height_hint=240,
            cols_hint=12,
        )

    def _refresh_pid_combo(self):
        pids = [
            str(pid)
            for pid in sorted(self.os.pcbs.keys())
            if self.os.pcbs[pid].state != STATE_TERMINATED
        ]
        cur = self.cmb_pid.get()
        self.cmb_pid["values"] = pids
        if cur not in pids:
            self.cmb_pid.set(pids[0] if pids else "")

    def _refresh_pagetable_text(self):
        self.txt_pt.delete("1.0", "end")
        pid_s = self.cmb_pid.get().strip()
        if not pid_s:
            self.txt_pt.insert("1.0", "无可用进程（或全部终止）\n")
            return
        pid = int(pid_s)
        pcb = self.os.pcbs.get(pid)
        if pcb is None:
            return
        lines = []
        lines.append(
            f"PID={pid} name={pcb.name} state={pcb.state} Q={pcb.queue_level} faults={pcb.page_faults}\n"
        )
        lines.append(
            f"mem={pcb.mem_kb}KB page={pcb.page_size_kb}KB pages={pcb.num_pages} | RAM固定块={pcb.ram_quota}\n"
        )
        lines.append(f"RAM pool frames: {pcb.ram_pool}\n")
        lines.append(f"RAM FIFO order : {list(pcb.ram_fifo)}\n\n")
        lines.append("VPN : RAM(present->frame) , SWAP(frame)\n")
        present_cnt = 0
        for vpn in range(pcb.num_pages):
            ent = pcb.page_table[vpn]
            if ent.present:
                present_cnt += 1
            swap_id = pcb.swap_map.get(vpn, "-")
            ram_part = f"{1 if ent.present else 0}->{ent.frame_id if ent.frame_id is not None else '-'}"
            lines.append(f"{vpn:03d} : RAM {ram_part} , SWAP {swap_id}\n")
        lines.append(
            f"\n驻留RAM页数：{present_cnt}/{pcb.num_pages}（不会超过固定块数 {pcb.ram_quota}）\n"
        )
        self.txt_pt.insert("1.0", "".join(lines))

    def _refresh_fs_tree(self):
        old_sel_path = self._get_selected_fs_path()
        self.tree_fs.delete(*self.tree_fs.get_children())
        self._tree_id_to_path.clear()

        root_id = self.tree_fs.insert("", "end", text="/", open=True)
        self._tree_id_to_path[root_id] = "/"

        def add_children(parent_id: str, node: FSNode):
            children = sorted(
                node.children.values(), key=lambda x: (not x.is_dir, x.name.lower())
            )
            for ch in children:
                if ch.is_dir:
                    label = f"{ch.name}/"
                else:
                    label = f"{ch.name} ({len(ch.content)}B)"
                item_id = self.tree_fs.insert(parent_id, "end", text=label, open=False)
                self._tree_id_to_path[item_id] = ch.path()
                if ch.is_dir:
                    add_children(item_id, ch)

        add_children(root_id, self.fs.root)

        if old_sel_path:
            for item_id, p in self._tree_id_to_path.items():
                if p == old_sel_path:
                    self.tree_fs.selection_set(item_id)
                    self.tree_fs.see(item_id)
                    self.tree_fs.focus(item_id)
                    break
        self.on_fs_select()

    def _refresh_file_combo(self):
        files = self.fs.list_file_paths()
        cur = self.cmb_file_run.get()
        self.cmb_file_run["values"] = files
        if cur not in files:
            self.cmb_file_run.set(files[0] if files else "")

    def _refresh_log(self):
        self.txt_log.delete("1.0", "end")
        self.txt_log.insert("1.0", "\n".join(self.os.log_lines))


def main():
    root = tk.Tk()
    root.geometry("1360x940")
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    MiniOSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
