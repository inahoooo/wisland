# -*- coding: utf-8 -*-
"""Agent 提醒：监听 DeepSeek Harness（DSH）的会话日志，在任务完成 / 等待授权时提醒。

DSH 把每个会话写成追加式的 zstd 分帧 JSONL：

    %USERPROFILE%/.dsh/sessions/--<cwd slug>--/<session id>/session.v4.jsonl.zstd

每帧是一次 append 批次，帧内可能有多行事件。我们只关心其中四种：

    turn/end          {turn, reason:{kind}}     任务结束（completed / error / aborted / …）
    approval/asked    {id, toolName}            弹出了审批，等你点
    approval/decided  {id, outcome}             审批已答复
    session/title     {title}                   会话标题

为什么读日志而不是别的路子（详见 HANDOFF 新章节）：DSH 不发 Windows toast，
它的 HTTP API 要会话令牌（GET / 直接 401），而投影缓存里的 approvalState 只有
{policy, turnOpen}、没有「待决审批」列表，做不出「正在等你批准哪个工具」。

读取是**纯增量**的：只按完整帧推进 offset，崩溃残留的半帧既不报错也不重复消费。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from pathlib import Path

from PySide6 import QtCore

from ..core.bg import run_bg, ui
from ..core.config import cfg, expand

try:
    import zstandard as zstd
except Exception:  # noqa: BLE001
    zstd = None

DSH_HOME = Path(os.path.expandvars(r"%USERPROFILE%\.dsh"))
SESSIONS_DIR = DSH_HOME / "sessions"
PROJCACHE_DIR = DSH_HOME / "storages" / "session_projcache" / "sessions"

DETAIL_MAX = 40          # 与 notify.toast_first_text 的 40 字上限保持一致
STATE_TTL_S = 7 * 86400  # 这么久没动静的会话状态就从内存里丢掉

_GEN_RE = re.compile(r"^session\.v(\d+)\.jsonl(\.zstd)?$")

# turn/end 的 reason.kind -> (状态, 文案)
_END_REASON = {
    "completed": ("done", "任务完成"),
    "max-tokens": ("error", "输出被截断"),
    "error": ("error", "出错中断"),
    "aborted": ("error", "已中止"),
    "interrupted": ("error", "已中断"),
    "blocked": ("error", "被阻塞"),
    "forked": ("error", "已派生分支"),
}


def _clip(s: str) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= DETAIL_MAX else s[: DETAIL_MAX - 1] + "…"


def _session_log(d: Path) -> Path | None:
    """会话目录里版本号最高的那份日志（兼容 compression: none 的明文行）。"""
    best: tuple[int, Path] | None = None
    try:
        for e in os.scandir(d):
            if not e.is_file():
                continue
            m = _GEN_RE.match(e.name)
            if not m:
                continue
            v = int(m.group(1))
            if best is None or v > best[0]:
                best = (v, Path(e.path))
    except OSError:
        return None
    return best[1] if best else None


class _Sess:
    """单个会话的增量读取状态。"""

    __slots__ = ("id", "path", "offset", "size", "depth", "cwds", "title",
                 "pending", "last_turn", "seen_at", "ready")

    def __init__(self, sid: str, path: Path):
        self.id = sid
        self.path = path
        self.offset = 0
        self.size = 0
        self.depth = 0
        self.cwds = ""
        self.title = ""
        self.pending: dict[str, dict] = {}
        self.last_turn = -1
        self.seen_at = time.time()
        self.ready = False       # 是否已经确定基线


class AgentWatchService(QtCore.QObject):
    """轮询 DSH 会话日志，把 turn/end 与审批事件翻译成消息页的「Agent」条目。"""

    event = QtCore.Signal(dict)   # {kind:'agent', status, detail, session, title, time, [quiet]}

    def __init__(self, parent=None):
        super().__init__(parent)
        ui(lambda: None)          # 让 bg 的 invoker 在主线程创建（否则 on_done 回调发不回来）
        self._t0 = time.time()    # 本进程启动时刻：早于它的会话一律基线化，不补发历史
        self._sessions: dict[str, _Sess] = {}
        self._lock = threading.Lock()     # _scan 在后台线程写 _sessions，主线程要读
        self._busy = False
        self._running = False
        self._disabled = ""
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.poll)

    # ---------- lifecycle ----------
    def start(self):
        if not cfg.get("agent.enabled", True):
            self._disabled = "config.agent.enabled = false"
            return
        if zstd is None:
            self._disabled = "缺少 zstandard 依赖（pip install zstandard），无法解析会话日志"
            print("wisland agent: " + self._disabled)
            return
        root = Path(expand(cfg.get("agent.sessions_dir") or str(SESSIONS_DIR)))
        if not root.is_dir():
            self._disabled = f"会话目录不存在：{root}"
            print("wisland agent: " + self._disabled)
            return
        self._root = root
        self._running = True
        sec = max(1, float(cfg.get("agent.interval_s") or 2))
        self._timer.start(int(sec * 1000))
        self.poll()

    def stop(self):
        self._running = False
        self._timer.stop()

    # ---------- public ----------
    def open(self) -> None:
        """点开 Agent 磁贴：DSH 在跑就切窗口，否则启动它。"""
        path = _dsh_exe()
        if not path:
            return
        opener = getattr(self, "opener", None)
        if opener is not None:
            opener(expand(path))
        else:
            from ..core.bg import launch
            launch(expand(path))

    def poll(self):
        if self._busy or not self._running:
            return
        self._busy = True
        with self._lock:
            snapshot = list(self._sessions.values())
        run_bg(self._scan, snapshot,
               on_done=self._on_scanned,
               on_error=lambda e: (setattr(self, "_busy", False), traceback.print_exception(type(e), e, e.__traceback__)))

    # ---------- 扫描（后台线程） ----------
    def _scan(self, known: list[_Sess]) -> list[tuple]:
        """返回 [(事件字典, 需要更新的 (sess, offset))]；不在这里跑 Qt 的东西。"""
        root: Path = getattr(self, "_root", SESSIONS_DIR)
        out: list[tuple] = []
        by_id = {s.id: s for s in known}

        for proj in _dirs(root):
            for sd in _dirs(proj):
                sid = sd.name
                path = _session_log(sd)
                if path is None:
                    continue
                st = by_id.get(sid)
                if st is None or st.path != path:
                    st = _Sess(sid, path)
                    with self._lock:
                        self._sessions[sid] = st
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < st.offset:          # 被截断 / 重写：重新基线化，不补发
                    st.offset = size
                # offset 没追平就一定要再读一次（撕裂帧之后长度可能不变）
                if size == st.size and st.ready and st.offset >= size:
                    continue
                st.size = size
                if not st.ready:
                    self._baseline(st, size)
                    if st.offset >= size:
                        continue
                try:
                    st.seen_at = time.time()
                    out.extend(self._read(st))
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        return out

    def _baseline(self, st: _Sess, size: int) -> None:
        """首次见到会话：读 header，决定从头读还是跳到末尾。"""
        st.ready = True
        head = _first_obj(st.path)
        st.depth = int(head.get("delegationDepth") or 0)
        st.cwds = str(head.get("cwd") or "")
        st.title = self._title_of(st.id)
        created = head.get("createdAt")
        try:
            created_s = float(created) / 1000.0 if created else 0.0
        except (TypeError, ValueError):
            created_s = 0.0
        # 本进程启动之后才创建的会话 = 真正的新会话，从头读；更早的一律跳过历史
        st.offset = 0 if created_s >= self._t0 else size

    def _read(self, st: _Sess) -> list[tuple]:
        """从 offset 读新帧，只按完整帧推进；返回事件并就地更新 st。"""
        with open(st.path, "rb") as f:
            f.seek(st.offset)
            data = f.read()
        if not data:
            return []

        text = ""
        pos = st.offset
        buf = data
        packed = st.path.suffix == ".zstd"
        while buf:
            if not packed:
                text += buf.decode("utf-8", "replace")
                pos += len(buf)
                buf = b""
                break
            dobj = zstd.ZstdDecompressor().decompressobj()
            try:
                chunk = dobj.decompress(buf)
            except Exception:  # noqa: BLE001  损坏帧：当撕裂处理，不推进
                break
            if not dobj.eof:
                break                       # 末尾半帧：整帧丢弃，下次从头解
            pos += len(buf) - len(dobj.unused_data)
            text += chunk.decode("utf-8", "replace")
            buf = dobj.unused_data

        st.offset = pos
        return self._translate(st, text)

    # ---------- 事件翻译 ----------
    def _translate(self, st: _Sess, text: str) -> list[tuple]:
        if not text:
            return []
        if st.depth > 0:                     # 子代理会话不提醒
            return []
        only = cfg.get("agent.only_cwds") or []
        if only and not any(st.cwds.lower().startswith(str(c).lower()) for c in only):
            return []

        events = []
        want = {"turn/end", "approval/asked", "approval/decided", "session/title"}
        for line in text.splitlines():
            line = line.strip()
            if not line or not any(k in line for k in want):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            typ = obj.get("type")
            if typ not in want:
                continue
            data = obj.get("data") or {}
            if typ == "session/title":
                t = str(data.get("title") or "").strip()
                if t:
                    st.title = t
                continue
            if typ == "turn/end":
                ev = self._on_turn_end(st, data)
            elif typ == "approval/asked":
                ev = self._on_asked(st, data)
            else:
                ev = self._on_decided(st, data)
            if ev:
                events.append((ev, 0))
        return events

    def _on_turn_end(self, st: _Sess, data: dict) -> dict | None:
        turn = data.get("turn")
        try:
            turn = int(turn)
        except (TypeError, ValueError):
            turn = None
        if turn is not None:
            if turn <= st.last_turn:          # 同一回合只发一次
                return None
            st.last_turn = turn
        kind = ((data.get("reason") or {}).get("kind")) or "completed"
        status, detail = _END_REASON.get(str(kind), ("done", "任务完成"))
        if status == "done" and not cfg.get("agent.notify_done", False):
            return None
        if status == "error" and not cfg.get("agent.notify_error", True):
            return None
        return self._mk(st, status, detail)

    def _on_asked(self, st: _Sess, data: dict) -> dict | None:
        rid = str(data.get("id") or "")
        tool = str(data.get("toolName") or "工具")
        if rid:
            st.pending[rid] = {"tool": tool, "time": time.time()}
        if not cfg.get("agent.notify_approval", True):
            return None
        return self._mk(st, "approval", _clip(f"等待批准：{tool}"))

    def _on_decided(self, st: _Sess, data: dict) -> dict | None:
        rid = str(data.get("id") or "")
        if not rid or rid not in st.pending:
            return None
        st.pending.pop(rid, None)
        return self._mk(st, "approval-cleared", "")

    def _mk(self, st: _Sess, status: str, detail: str) -> dict:
        if not st.title:
            st.title = self._title_of(st.id) or Path(st.cwds).name or "Agent"
        d = {"kind": "agent", "name": st.title, "source": "agent", "status": status,
             "detail": detail, "session": st.id[-8:], "time": time.time()}
        if cfg.get("agent.skip_foreground", True) and self._foreground():
            d["quiet"] = True
        return d

    # ---------- 辅助 ----------
    def _title_of(self, sid: str) -> str:
        f = PROJCACHE_DIR / f"{sid}.json"
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            return str(((j.get("record") or {}).get("rows") or {}).get("title", {}).get("val") or "")
        except (OSError, ValueError, AttributeError):
            return ""

    def _foreground(self) -> bool:
        """DSH 正在前台时不打扰（避免自己给自己弹提醒）。"""
        try:
            import ctypes
            from .notify import exe_of_hwnd
            hwnd = ctypes.WinDLL("user32").GetForegroundWindow()
            exe = exe_of_hwnd(int(hwnd))
            want = os.path.basename(_dsh_exe()).lower()
            return bool(exe and want and exe == want)
        except Exception:  # noqa: BLE001
            return False

    def _on_scanned(self, items):
        self._busy = False
        now = time.time()
        with self._lock:
            for sid in [k for k, v in self._sessions.items() if now - v.seen_at > STATE_TTL_S]:
                self._sessions.pop(sid, None)
        for ev, _ in items or []:
            self.event.emit(ev)


_dsh_cache: list[str] = []


def _dsh_exe() -> str:
    """DSH 可执行文件：优先用配置里的 agent.exe，否则从正在运行的进程里找。

    找不到就返回空串（点磁贴不动作），不影响提醒本身——提醒只依赖会话日志。
    """
    got = str(cfg.get("agent.exe") or "").strip()
    if got:
        return got
    if _dsh_cache:
        return _dsh_cache[0]
    for name in ("DeepSeek Harness.exe",):
        try:
            import psutil
            for p in psutil.process_iter(["name", "exe"]):
                if (p.info.get("name") or "").lower() == name.lower():
                    exe = p.info.get("exe") or ""
                    if exe:
                        _dsh_cache.append(exe)
                        return exe
        except Exception:  # noqa: BLE001
            break
    return ""


def _dirs(p: Path):
    try:
        return [Path(e.path) for e in os.scandir(p) if e.is_dir()]
    except OSError:
        return []


def _first_obj(path: Path) -> dict:
    """读日志第一帧里的会话 header（不是 zstd 就当第一行读）。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(65536)
        if path.suffix == ".zstd" and zstd is not None:
            dobj = zstd.ZstdDecompressor().decompressobj()
            raw = dobj.decompress(raw)
        line = raw.decode("utf-8", "replace").split("\n", 1)[0].strip()
        return json.loads(line) if line else {}
    except Exception:  # noqa: BLE001
        return {}
