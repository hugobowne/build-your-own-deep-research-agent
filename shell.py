import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from rich.console import RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static


SubmitResult: TypeAlias = Awaitable[None] | None
SubmitHandler: TypeAlias = Callable[[str], SubmitResult]
ReadyHandler: TypeAlias = Callable[[], SubmitResult]


class ShellApp(App[None]):
    CSS = """
    Screen {
        background: #0b0d12;
        color: #e5e7eb;
    }

    #shell {
        width: 100%;
        height: 100%;
        padding: 0 1 0 2;
        background: #0b0d12;
    }

    #transcript_scroll {
        height: 1fr;
        padding: 1 0 1 1;
        background: #0b0d12;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-background: #0f1117;
        scrollbar-color: #3b4457;
    }

    #transcript_flow {
        height: auto;
    }

    #regions {
        height: auto;
        margin-top: 1;
    }

    .entry {
        height: auto;
        margin-bottom: 1;
    }

    .region {
        height: auto;
        margin-bottom: 1;
        padding-left: 1;
        color: #dbe2ea;
    }

    #composer_wrap {
        height: auto;
        margin: 0 0 1 0;
        border-top: solid #202633;
        padding: 1 0 0 1;
    }

    #loading_label {
        height: auto;
        color: #94a3b8;
        margin-bottom: 0;
    }

    #composer_row {
        height: auto;
        layout: horizontal;
    }

    #composer_prompt {
        width: 2;
        color: #60a5fa;
        content-align: left middle;
    }

    #composer {
        height: 1;
        width: 1fr;
        min-height: 1;
        border: none;
        background: #0b0d12;
        color: #f3f4f6;
        padding: 0 0 0 1;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.on_submit: SubmitHandler | None = None
        self.on_ready: ReadyHandler | None = None
        self.submit_task: asyncio.Task[None] | None = None
        self.loading_task: asyncio.Task[None] | None = None
        self.transcript_scroll = VerticalScroll(id="transcript_scroll")
        self.transcript_flow = Vertical(id="transcript_flow")
        self.regions = Vertical(id="regions")
        self.loading_label = Static("", id="loading_label")
        self.composer_prompt = Static(">", id="composer_prompt")
        self.composer = Input(
            placeholder="Send a message. Enter submits.",
            id="composer",
        )
        self.entry_widgets: dict[str, Static] = {}
        self.region_widgets: dict[str, Static] = {}

    def compose(self) -> ComposeResult:
        with Container(id="shell"):
            with Vertical():
                with self.transcript_scroll:
                    yield self.transcript_flow
                yield self.regions
                with Container(id="composer_wrap"):
                    yield self.loading_label
                    with Horizontal(id="composer_row"):
                        yield self.composer_prompt
                        yield self.composer

    async def on_mount(self) -> None:
        self.composer.focus()
        if self.on_ready is not None:
            result = self.on_ready()
            if result is not None:
                await result

    async def _run_loading_dots(self) -> None:
        frames = [".", "..", "..."]
        index = 0
        try:
            while True:
                self.loading_label.update(frames[index % len(frames)])
                index += 1
                await asyncio.sleep(0.35)
        except asyncio.CancelledError:
            self.loading_label.update("")
            raise

    async def _append_item(self, content: RenderableType | object) -> None:
        widget = Static(content, classes="entry")
        await self.transcript_flow.mount(widget)
        self.transcript_scroll.scroll_end(animate=False)

    def write(self, content: RenderableType | object) -> None:
        self.run_worker(self._append_item(content), exclusive=False)

    def set_loading(self, is_loading: bool) -> None:
        self.composer.placeholder = (
            "Thinking..." if is_loading else "Send a message. Enter submits."
        )
        if is_loading:
            if self.loading_task is None or self.loading_task.done():
                self.loading_task = asyncio.create_task(self._run_loading_dots())
            return

        if self.loading_task is not None and not self.loading_task.done():
            self.loading_task.cancel()
        self.loading_task = None
        self.loading_label.update("")

    async def ensure_region(self, name: str) -> Static:
        existing = self.region_widgets.get(name)
        if existing is not None:
            return existing

        widget = Static(id=f"region-{name}", classes="region")
        self.region_widgets[name] = widget
        await self.regions.mount(widget)
        return widget

    async def update_region(self, name: str, content: RenderableType | object) -> None:
        widget = await self.ensure_region(name)
        widget.update(content)

    def clear_region(self, name: str) -> None:
        widget = self.region_widgets.get(name)
        if widget is not None:
            widget.update(Text(""))

    async def ensure_entry(self, name: str) -> Static:
        existing = self.entry_widgets.get(name)
        if existing is not None:
            return existing

        widget = Static(id=f"entry-{name}", classes="entry")
        self.entry_widgets[name] = widget
        await self.transcript_flow.mount(widget)
        self.transcript_scroll.scroll_end(animate=False)
        return widget

    async def update_entry(self, name: str, content: RenderableType | object) -> None:
        widget = await self.ensure_entry(name)
        widget.update(content)
        self.transcript_scroll.scroll_end(animate=False)

    def clear_entry(self, name: str) -> None:
        widget = self.entry_widgets.get(name)
        if widget is not None:
            widget.update(Text(""))

    def set_submit_handler(self, handler: SubmitHandler) -> None:
        self.on_submit = handler

    def set_ready_handler(self, handler: ReadyHandler) -> None:
        self.on_ready = handler

    async def _run_submit_handler(self, text: str) -> None:
        try:
            if self.on_submit is None:
                return
            result = self.on_submit(text)
            if result is not None:
                await result
        except Exception as exc:
            self.write(f"[bold red]Error[/bold red]\n{exc}")
        finally:
            self.submit_task = None

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text in {"/exit", "exit", "quit"}:
            self.exit()
            return
        if self.submit_task is not None and not self.submit_task.done():
            self.bell()
            return

        self.composer.value = ""
        self.submit_task = asyncio.create_task(self._run_submit_handler(text))


class Shell:
    def __init__(self) -> None:
        self.app = ShellApp()
        self._pending_prints: list[RenderableType | object] = []
        self._pending_entries: list[tuple[str, RenderableType | object]] = []
        self._pending_regions: list[tuple[str, RenderableType | object]] = []
        self._pending_loading = False
        self._initialized = False

    def initialize(
        self,
        *,
        on_submit: SubmitHandler | None = None,
        on_ready: ReadyHandler | None = None,
    ) -> None:
        if on_submit is not None:
            self.app.set_submit_handler(on_submit)
        if on_ready is not None:
            self.app.set_ready_handler(on_ready)
        self._initialized = True

    def _call_in_app(self, callback: Callable[[], None]) -> None:
        app_thread_id = getattr(self.app, "_thread_id", None)
        if self.app.is_running and app_thread_id is not None and app_thread_id != threading.get_ident():
            self.app.call_from_thread(callback)
        else:
            callback()

    def print(self, content: RenderableType | object) -> None:
        if not self.app.is_running:
            self._pending_prints.append(content)
            return
        self._call_in_app(lambda: self.app.write(content))

    def set_loading(self, is_loading: bool) -> None:
        if not self.app.is_running:
            self._pending_loading = is_loading
            return
        self._call_in_app(lambda: self.app.set_loading(is_loading))

    def update_region(self, name: str, content: RenderableType | object) -> None:
        if not self.app.is_running:
            self._pending_regions.append((name, content))
            return

        def update() -> None:
            self.app.run_worker(self.app.update_region(name, content), exclusive=False)

        self._call_in_app(update)

    def update_entry(self, name: str, content: RenderableType | object) -> None:
        if not self.app.is_running:
            self._pending_entries.append((name, content))
            return

        def update() -> None:
            self.app.run_worker(self.app.update_entry(name, content), exclusive=False)

        self._call_in_app(update)

    def clear_region(self, name: str) -> None:
        self._call_in_app(lambda: self.app.clear_region(name))

    def clear_entry(self, name: str) -> None:
        self._call_in_app(lambda: self.app.clear_entry(name))

    async def _flush_pending(self) -> None:
        self.app.set_loading(self._pending_loading)
        for content in self._pending_prints:
            self.app.write(content)
        self._pending_prints.clear()
        for name, content in self._pending_entries:
            await self.app.update_entry(name, content)
        self._pending_entries.clear()
        for name, content in self._pending_regions:
            await self.app.update_region(name, content)
        self._pending_regions.clear()

    def run(self) -> None:
        if not self._initialized:
            self.initialize()

        original_ready = self.app.on_ready

        async def ready_wrapper() -> None:
            await self._flush_pending()
            if original_ready is not None:
                result = original_ready()
                if result is not None:
                    await result

        self.app.set_ready_handler(ready_wrapper)
        self.app.run()
