"""
Lightweight HTTP server that exposes the MiniOS simulation as JSON plus a
browser-friendly dashboard (served from ./web).

Run:
    python web_server.py --port 8000
Then open http://localhost:8000 in your browser.
"""

import argparse
import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from test import (
    DEFAULT_MEM_SIZE_KB,
    DEFAULT_PAGE_SIZE_KB,
    DEFAULT_PROC_MEM_KB,
    DEFAULT_PROC_WORK,
    DEFAULT_VM_SIZE_KB,
    FileSystem,
    MiniOS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")


class OSContainer:
    """Holds the MiniOS instance and a lock for thread-safe mutations."""

    def __init__(self) -> None:
        self.fs = FileSystem()
        self.os = MiniOS(
            DEFAULT_PAGE_SIZE_KB, DEFAULT_MEM_SIZE_KB, DEFAULT_VM_SIZE_KB, fs=self.fs
        )
        self.lock = threading.RLock()


STATE = OSContainer()


# ---------------------- Serialization helpers ---------------------- #


def _page_table_rows(pcb) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for vpn in range(pcb.num_pages):
        ent = pcb.page_table[vpn]
        rows.append(
            {
                "vpn": vpn,
                "present": ent.present,
                "frame": ent.frame_id,
                "swap": pcb.swap_map.get(vpn),
            }
        )
    return rows


def _pcb_to_dict(pcb) -> Dict[str, Any]:
    return {
        "pid": pcb.pid,
        "name": pcb.name,
        "state": pcb.state,
        "queue": pcb.queue_level,
        "slice_left": pcb.slice_left,
        "pc": pcb.pc,
        "remaining": pcb.remaining,
        "page_faults": pcb.page_faults,
        "block_left": pcb.block_left,
        "mem_kb": pcb.mem_kb,
        "num_pages": pcb.num_pages,
        "ram_quota": pcb.ram_quota,
        "ram_pool": list(pcb.ram_pool),
        "ram_fifo": list(pcb.ram_fifo),
        "page_table": _page_table_rows(pcb),
    }


def _fs_tree(node) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": node.name or "/",
        "path": node.path(),
        "is_dir": node.is_dir,
    }
    if node.is_dir:
        children = sorted(
            node.children.values(), key=lambda x: (not x.is_dir, x.name.lower())
        )
        info["children"] = [_fs_tree(ch) for ch in children]
    else:
        info["size"] = len(node.content)
    return info


def _ram_snapshot(os_obj: MiniOS) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for fr in os_obj.ram.frames:
        data.append(
            {
                "id": fr.frame_id,
                "reserved": fr.reserved_pid,
                "owner": {"pid": fr.owner[0], "vpn": fr.owner[1]} if fr.owner else None,
            }
        )
    return data


def _swap_snapshot(os_obj: MiniOS) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for fr in os_obj.swap.frames:
        data.append(
            {
                "id": fr.frame_id,
                "owner": {"pid": fr.owner[0], "vpn": fr.owner[1]} if fr.owner else None,
            }
        )
    return data


def build_snapshot() -> Dict[str, Any]:
    """Grab a point-in-time view of the OS state."""
    with STATE.lock:
        os_obj = STATE.os
        procs = [_pcb_to_dict(os_obj.pcbs[pid]) for pid in sorted(os_obj.pcbs.keys())]
        return {
            "ticks": os_obj.ticks,
            "config": {
                "page_kb": os_obj.page_size_kb,
                "mem_kb": os_obj.mem_size_kb,
                "vm_kb": os_obj.vm_size_kb,
                "ram_frames": os_obj.ram_frames,
                "swap_frames": os_obj.swap_frames,
            },
            "running_pid": os_obj.sched.running_pid,
            "queues": os_obj.sched.snapshot_queues(),
            "processes": procs,
            "ram": _ram_snapshot(os_obj),
            "swap": _swap_snapshot(os_obj),
            "fs_tree": _fs_tree(os_obj.fs.root),
            "files": os_obj.fs.list_file_paths(),
            "log": os_obj.log_lines[-500:],
        }


# --------------------------- HTTP layer --------------------------- #


class OSRequestHandler(SimpleHTTPRequestHandler):
    """Serves static files under ./web and handles /api/* JSON requests."""

    def __init__(self, *args, directory: Optional[str] = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        # Keep console noise low; override if you want verbose access logs.
        print("[http]", fmt % args)

    # CORS / preflight
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._write_cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            return self._send_json(build_snapshot())
        if self.path.startswith("/api/fs/read"):
            return self._handle_fs_read()
        return super().do_GET()

    def do_POST(self) -> None:
        if not self.path.startswith("/api/"):
            return self._send_json({"error": "Unknown endpoint"}, status=404)

        data = self._read_json_body()
        try:
            if self.path == "/api/init":
                return self._handle_init(data)
            if self.path == "/api/tick":
                return self._handle_tick(data)
            if self.path == "/api/create":
                return self._handle_create(data)
            if self.path == "/api/run_file":
                return self._handle_run_file(data)
            if self.path == "/api/kill":
                return self._handle_kill(data)
            if self.path == "/api/fs/mkdir":
                return self._handle_fs_mkdir(data)
            if self.path == "/api/fs/touch":
                return self._handle_fs_touch(data)
            if self.path == "/api/fs/write":
                return self._handle_fs_write(data)
            if self.path == "/api/seed_demo":
                return self._handle_seed_demo()
        except Exception as exc:  # pragma: no cover - error surface
            return self._send_json({"error": str(exc)}, status=400)

        return self._send_json({"error": "Unknown endpoint"}, status=404)

    # -------- API handlers -------- #
    def _handle_init(self, data: Dict[str, Any]) -> None:
        page_kb = int(data.get("page_kb", DEFAULT_PAGE_SIZE_KB))
        mem_kb = int(data.get("mem_kb", DEFAULT_MEM_SIZE_KB))
        vm_kb = int(data.get("vm_kb", DEFAULT_VM_SIZE_KB))
        if page_kb <= 0 or mem_kb <= 0 or vm_kb <= 0:
            return self._send_json({"error": "All sizes must be positive"}, status=400)
        if mem_kb % page_kb != 0 or vm_kb % page_kb != 0:
            return self._send_json(
                {"error": "Memory and VM sizes must be divisible by page size"},
                status=400,
            )

        with STATE.lock:
            STATE.os = MiniOS(page_kb, mem_kb, vm_kb, fs=STATE.fs)
        return self._send_json(build_snapshot())

    def _handle_tick(self, data: Dict[str, Any]) -> None:
        count = int(data.get("count", 1) or 1)
        count = max(1, min(count, 50))
        with STATE.lock:
            for _ in range(count):
                STATE.os.tick()
        return self._send_json(build_snapshot())

    def _handle_create(self, data: Dict[str, Any]) -> None:
        name = (data.get("name") or "P").strip() or "P"
        mem_kb = int(data.get("mem_kb", DEFAULT_PROC_MEM_KB))
        work = int(data.get("work", DEFAULT_PROC_WORK))
        with STATE.lock:
            STATE.os.create_process(name=name, mem_kb=mem_kb, work_len=work)
        return self._send_json(build_snapshot())

    def _handle_run_file(self, data: Dict[str, Any]) -> None:
        path = (data.get("path") or "").strip()
        if not path:
            return self._send_json({"error": "path is required"}, status=400)
        nprocs = int(data.get("nprocs", 2))
        mem_each = int(data.get("mem_kb_each", DEFAULT_PROC_MEM_KB))
        with STATE.lock:
            STATE.os.run_file_spawn_processes(path, nprocs=nprocs, mem_kb_each=mem_each)
        return self._send_json(build_snapshot())

    def _handle_kill(self, data: Dict[str, Any]) -> None:
        pid = int(data.get("pid", -1))
        if pid <= 0:
            return self._send_json({"error": "pid must be > 0"}, status=400)
        with STATE.lock:
            STATE.os.kill_process(pid)
        return self._send_json(build_snapshot())

    def _handle_seed_demo(self) -> None:
        with STATE.lock:
            fs = STATE.fs
            os_obj = STATE.os
            for name in ("docs", "bin", "data"):
                if not fs.exists(f"/{name}"):
                    fs.mkdir("/", name)
            if not fs.exists("/docs/algo.txt"):
                fs.touch("/docs", "algo.txt")
                fs.write(
                    "/docs/algo.txt",
                    "MLFQ scheduling with Local FIFO paging & fixed resident frames.\n",
                    append=True,
                )
            if not fs.exists("/docs/hello.txt"):
                fs.touch("/docs", "hello.txt")
                fs.write(
                    "/docs/hello.txt", "Hello OS! Paging & VM demo.\n", append=True
                )
            try:
                os_obj.run_file_spawn_processes(
                    "/docs/algo.txt", nprocs=3, mem_kb_each=min(32, DEFAULT_PROC_MEM_KB)
                )
            except Exception:
                pass
        return self._send_json(build_snapshot())

    def _handle_fs_mkdir(self, data: Dict[str, Any]) -> None:
        dir_path = data.get("dir_path") or "/"
        name = data.get("name") or ""
        with STATE.lock:
            new_path = STATE.fs.mkdir(dir_path, name)
            STATE.os.log(f"fs: mkdir {new_path}")
        return self._send_json(build_snapshot())

    def _handle_fs_touch(self, data: Dict[str, Any]) -> None:
        dir_path = data.get("dir_path") or "/"
        name = data.get("name") or ""
        with STATE.lock:
            new_path = STATE.fs.touch(dir_path, name)
            STATE.os.log(f"fs: touch {new_path}")
        return self._send_json(build_snapshot())

    def _handle_fs_write(self, data: Dict[str, Any]) -> None:
        path = data.get("path") or ""
        text = data.get("text") or ""
        append = bool(data.get("append", True))
        if not path:
            return self._send_json({"error": "path is required"}, status=400)
        with STATE.lock:
            STATE.fs.write(path, text, append=append)
            STATE.os.log(
                f"fs: {'append' if append else 'overwrite'} write {path} bytes={len(text)}"
            )
        return self._send_json(build_snapshot())

    def _handle_fs_read(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        path = qs.get("path", [None])[0]
        if not path:
            return self._send_json({"error": "path is required"}, status=400)
        with STATE.lock:
            content = STATE.fs.read(path)
        return self._send_json({"path": path, "content": content})

    # -------- Helpers -------- #
    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._write_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the MiniOS web dashboard (static files + JSON API)."
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    if not os.path.isdir(WEB_DIR):
        raise SystemExit("Missing ./web directory; create it before running the server.")

    handler = partial(OSRequestHandler, directory=WEB_DIR)
    server = ThreadingHTTPServer(("", args.port), handler)
    print(f"MiniOS web UI available at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
