"""统一进度条与回调（单条显示，子步骤通过回调上报）"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar

T = TypeVar("T")

ProgressCallback = Callable[["ProgressUpdate"], None]

_BAR_FMT = "{desc} | {postfix} [{elapsed}]"


@dataclass
class ProgressUpdate:
    """进度事件，供 UI 与外部持久化使用"""

    stage: str
    current: Optional[int] = None
    total: Optional[int] = None
    message: str = ""
    done: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


def format_progress_line(u: ProgressUpdate) -> str:
    parts = [u.stage]
    if u.total and u.total > 0 and u.current is not None:
        pct = min(100.0, 100.0 * u.current / u.total)
        parts.append(f"{u.current}/{u.total} ({pct:.0f}%)")
    elif u.message:
        parts.append(u.message)
    return " · ".join(parts)


class ProgressReporter:
    """单条 tqdm + 可选回调；阻塞阶段无 total 时仍刷新 elapsed"""

    def __init__(
        self,
        *,
        show: bool = True,
        desc: str = "HierAD",
        callback: Optional[ProgressCallback] = None,
    ):
        self.show = show
        self.desc = desc
        self.callback = callback
        self._bar = None
        self._stage_name = ""
        self._stage_current = 0
        self._stage_total: Optional[int] = None
        self._stop = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        if show:
            self._init_bar()

    def _init_bar(self) -> None:
        try:
            from tqdm import tqdm

            self._bar = tqdm(
                total=None,
                desc=self.desc,
                bar_format=_BAR_FMT,
                mininterval=0.3,
                leave=True,
            )
        except ImportError:
            self._bar = None

    def _start_refresh(self) -> None:
        if self._bar is None:
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(0.5):
                if self._bar is not None:
                    self._bar.refresh()

        self._refresh_thread = threading.Thread(target=_loop, daemon=True)
        self._refresh_thread.start()

    def _stop_refresh(self) -> None:
        self._stop.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=1.0)
            self._refresh_thread = None

    def _emit(self, update: ProgressUpdate) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(format_progress_line(update), refresh=True)
        if self.callback:
            self.callback(update)

    def stage(
        self,
        name: str,
        *,
        total: Optional[int] = None,
        current: Optional[int] = None,
        message: str = "",
        done: bool = False,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name != self._stage_name:
            self._stop_refresh()
            self._stage_name = name
            self._stage_current = 0
            self._stage_total = total
            if total is None and self.show and not done:
                self._start_refresh()
        if total is not None:
            self._stage_total = total
        if current is not None:
            self._stage_current = current

        update = ProgressUpdate(
            stage=name,
            current=self._stage_current,
            total=self._stage_total,
            message=message,
            done=done,
            data=data or {},
        )
        self._emit(update)
        if done:
            self._stop_refresh()

    def advance(
        self,
        n: int = 1,
        *,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._stage_current += n
        self.stage(
            self._stage_name,
            current=self._stage_current,
            total=self._stage_total,
            message=message,
            data=data,
        )

    def update(
        self,
        current: int,
        total: Optional[int] = None,
        *,
        message: str = "",
        done: bool = False,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stage(
            self._stage_name or "进度",
            total=total if total is not None else self._stage_total,
            current=current,
            message=message,
            done=done,
            data=data,
        )

    def close(self) -> None:
        self._stop_refresh()
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def progress_iter(
    iterable: Iterable[T],
    *,
    show: bool = False,
    reporter: Optional[ProgressReporter] = None,
    stage: Optional[str] = None,
    desc: str = "",
    unit: str = "it",
    total: Optional[int] = None,
    leave: bool = True,
    position: Optional[int] = None,
) -> Iterable[T]:
    """迭代进度：优先 reporter（不嵌套第二条 bar），否则可选 legacy tqdm"""
    if reporter is not None:
        stage_name = stage or desc or reporter._stage_name or "进度"
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        reporter.stage(stage_name, total=total, current=0)
        for item in iterable:
            yield item
            reporter.advance()
        reporter.stage(stage_name, current=total, total=total, done=True)
        return

    if not show:
        yield from iterable
        return

    try:
        from tqdm import tqdm

        yield from tqdm(
            iterable,
            desc=desc,
            unit=unit,
            total=total,
            leave=leave,
            position=position,
        )
    except ImportError:
        yield from iterable


@contextmanager
def progress_scope(
    reporter: Optional[ProgressReporter],
    stage: str,
    *,
    total: Optional[int] = None,
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
):
    """阻塞阶段：进入上报 stage，退出标记 done"""
    noop = ProgressReporter(show=False)
    rep = reporter or noop
    rep.stage(stage, total=total, message=message, data=data or {})
    try:
        yield rep
    finally:
        rep.stage(
            stage,
            current=rep._stage_current,
            total=rep._stage_total,
            done=True,
            data=data or {},
        )


def progress_task(*, show: bool, desc: str, **kwargs):
    from contextlib import nullcontext

    if not show:
        return nullcontext()

    class _Scope:
        def __init__(self):
            self.reporter = ProgressReporter(show=True, desc=desc)

        def __enter__(self):
            self.reporter.stage(desc)
            return self.reporter

        def __exit__(self, *args):
            self.reporter.stage(desc, done=True)

    return _Scope()


def blocking_spinner(*, show: bool, desc: str, hint: str = "", **kwargs):
    return progress_task(show=show, desc=desc)


def progress_steps(steps: list[str], *, show: bool, desc: str = "进度", **kwargs):
    """兼容旧 API"""
    if not show:
        return lambda _label: None
    rep = ProgressReporter(show=True, desc=desc)
    rep.stage(steps[0] if steps else desc)

    def tick(label: str) -> None:
        rep.stage(label, done=True)
        if label in steps:
            idx = steps.index(label)
            if idx + 1 < len(steps):
                rep.stage(steps[idx + 1])

    return tick
