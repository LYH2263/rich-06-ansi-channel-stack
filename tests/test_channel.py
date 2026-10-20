import io
import logging
import threading

import pytest

from rich.channel import (
    DEFAULT_CHANNEL_POLICY,
    ChannelPolicy,
    ChannelStack,
    ChannelStackError,
)
from rich.console import Console, Group
from rich.control import Control
from rich.file_proxy import FileProxy
from rich.jupyter import _render_segments
from rich.live import Live
from rich.segment import ControlType, Segment
from rich.text import Text


def make_console(
    *, force_terminal=True, color_system=None, record=False, file=None
) -> Console:
    return Console(
        file=file or io.StringIO(),
        force_terminal=force_terminal,
        legacy_windows=False,
        color_system=color_system,
        record=record,
        _environ={},
    )


# --------------------------------------------------------------------------
# Stack basics
# --------------------------------------------------------------------------


def test_default_policy():
    console = make_console(force_terminal=False)
    assert console.channel_policy == ChannelPolicy(
        style=True, control=False, title=False
    )
    assert console.channel_policy is DEFAULT_CHANNEL_POLICY


def test_stack_push_pop():
    stack = ChannelStack()
    policy = ChannelPolicy(control=True)
    assert stack.top is DEFAULT_CHANNEL_POLICY
    stack.push_policy(policy)
    assert stack.top is policy
    stack.push_policy(ChannelPolicy(title=True))
    assert stack.top == ChannelPolicy(title=True)
    stack.pop_policy()
    assert stack.top is policy
    stack.pop_policy()
    assert stack.top is DEFAULT_CHANNEL_POLICY


def test_stack_pop_empty_raises():
    stack = ChannelStack()
    with pytest.raises(ChannelStackError):
        stack.pop_policy()


def test_context_pops_on_exception():
    console = make_console(force_terminal=False)
    policy = ChannelPolicy(control=True)
    with pytest.raises(ValueError):
        with console.use_channel_policy(policy):
            assert console.channel_policy is policy
            raise ValueError("boom")
    assert console.channel_policy is DEFAULT_CHANNEL_POLICY


def test_stacks_cross_nest_independently():
    from rich.theme import Theme

    console = make_console(force_terminal=False)
    policy = ChannelPolicy(control=True)
    theme = Theme({"foo": "red"}, inherit=False)

    console.push_channel_policy(policy)
    console.push_theme(theme)
    try:
        assert console.channel_policy is policy
        # Pop in an arbitrary interleaved order; each stack is independent.
        console.pop_channel_policy()
        assert console.channel_policy is DEFAULT_CHANNEL_POLICY
        with pytest.raises(ChannelStackError):
            console.pop_channel_policy()
    finally:
        console.pop_theme()
    assert console.channel_policy is DEFAULT_CHANNEL_POLICY


def test_channel_stack_is_thread_local():
    console = make_console(force_terminal=False)
    seen = []

    def read_policy():
        seen.append(console.channel_policy)

    with console.use_channel_policy(ChannelPolicy(control=True)):
        thread = threading.Thread(target=read_policy)
        thread.start()
        thread.join()

    assert seen == [DEFAULT_CHANNEL_POLICY]


# --------------------------------------------------------------------------
# FileProxy: default behaviour (style only)
# --------------------------------------------------------------------------


def test_file_proxy_default_style_only_terminal():
    file = io.StringIO()
    console = make_console(force_terminal=True, color_system="truecolor", file=file)
    proxy = FileProxy(console, file)
    proxy.write("\x1b[31mred\x1b[0m\x1b[1A\x07\x1b]0;title\x07tail\n")
    output = file.getvalue()
    # SGR survives
    assert "\x1b[31mred\x1b[0m" in output
    # mappable CSI, bell and title OSC are dropped, not written as characters
    assert "\x1b[1A" not in output
    assert "\x07" not in output
    assert "title" not in output
    assert output.endswith("tail\n")


def test_file_proxy_default_plain_when_not_terminal():
    file = io.StringIO()
    console = make_console(force_terminal=False, file=file)
    proxy = FileProxy(console, file)
    proxy.write("\x1b[31mred\x1b[0m\x1b[2Kx\n")
    assert file.getvalue() == "redx\n"


def test_decoded_text_is_not_markup_or_highlighted():
    file = io.StringIO()
    console = make_console(force_terminal=True, file=file)
    proxy = FileProxy(console, file)
    proxy.write("[not_markup] http://example.org end\n")
    assert file.getvalue() == "[not_markup] http://example.org end\n"


# --------------------------------------------------------------------------
# control channel
# --------------------------------------------------------------------------


def test_control_channel_mapping_and_order():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(control=True)):
        proxy.write("a\x1b[2Ab\x1b[Cc\x1b[10Gd\x1b[Ke\x07f\n")
    assert file.getvalue() == ("a\x1b[2Ab\x1b[1Cc\x1b[10Gd\x1b[0Ke\x07f\n")


def test_control_channel_all_directions():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(control=True)):
        proxy.write("\x1b[A\x1b[B\x1b[C\x1b[D\x1b[G\n")
    # Defaults: A-D default to 1; G column is 1 based inbound, Rich is 0 based
    assert file.getvalue() == "\x1b[1A\x1b[1B\x1b[1C\x1b[1D\x1b[1G\n"


def test_control_channel_closed_drops_without_literal_text():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    proxy.write("a\x1b[2Ab\x1b[2Kc\x07d\n")
    assert file.getvalue() == "abcd\n"


def test_unmappable_csi_always_dropped():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(control=True)):
        # private mode, HOME and erase display are not in the mapped set
        proxy.write("a\x1b[?25hb\x1b[Hc\x1b[2Jd\n")
    assert file.getvalue() == "abcd\n"


# --------------------------------------------------------------------------
# title channel
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence",
    [
        "\x1b]0;hello\x07",
        "\x1b]2;hello\x07",
        "\x1b]0;hello\x1b\\",
        "\x1b]2;hello\x1b\\",
    ],
)
def test_title_channel_open(sequence):
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(title=True)):
        proxy.write(sequence + "x\n")
    assert file.getvalue() == "\x1b]0;hello\x07x\n"


def test_title_channel_closed():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    proxy.write("\x1b]0;hello\x07x\x1b]2;world\x1b\\y\n")
    assert file.getvalue() == "xy\n"


def test_non_title_osc_never_emitted():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(control=True, title=True)):
        # OSC 1 (icon name) is not a window title sequence
        proxy.write("\x1b]1;icon\x07x\n")
    assert file.getvalue() == "x\n"


# --------------------------------------------------------------------------
# style channel state machine isolation
# --------------------------------------------------------------------------


def test_style_off_does_not_apply_or_leak_sgr():
    file = io.StringIO()
    console = make_console(color_system="truecolor", file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(style=False)):
        proxy.write("\x1b[31mred-while-off\x1b[1m\n")
    output = file.getvalue()
    assert "\x1b[" not in output
    assert output == "red-while-off\n"

    # Reopening style must not pick up state accumulated while closed.
    proxy.write("plain-after\n")
    reopened = file.getvalue()[len(output) :]
    assert reopened == "plain-after\n"
    assert "\x1b[31m" not in reopened
    assert "\x1b[1m" not in reopened


def test_style_state_before_close_survives_but_closed_codes_do_not():
    file = io.StringIO()
    console = make_console(color_system="truecolor", file=file)
    proxy = FileProxy(console, file)
    proxy.write("\x1b[31mred-line\n")
    with console.use_channel_policy(ChannelPolicy(style=False)):
        proxy.write("\x1b[1mbold-attempt\n")
    proxy.write("still-red-not-bold\n")
    output = file.getvalue()
    lines = output.splitlines()
    assert "bold-attempt" in lines and "\x1b[" not in lines[1]
    last_line = lines[2]
    assert "still-red-not-bold" in last_line
    assert "\x1b[31m" in last_line
    assert "\x1b[1m" not in last_line


def test_style_off_does_not_leak_hyperlink():
    file = io.StringIO()
    console = make_console(color_system="truecolor", file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(style=False)):
        proxy.write("\x1b]8;;http://example.org\x1b\\linked\x1b]8;;\x1b\\\n")
    proxy.write("after\n")
    output = file.getvalue()
    assert "http://example.org" not in output
    assert output.endswith("linked\nafter\n")


def test_style_off_control_on_same_line():
    file = io.StringIO()
    console = make_console(file=file)
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(style=False, control=True)):
        proxy.write("a\x1b[31mb\x1b[2Ac\x1b[1md\n")
    # Control emitted in order between text; style never enters the Text.
    assert file.getvalue() == "ab\x1b[2Acd\n"


# --------------------------------------------------------------------------
# Multiple proxies share the one stack top, read per write
# --------------------------------------------------------------------------


def test_proxies_share_stack_and_read_per_write():
    file = io.StringIO()
    console = make_console(file=file)
    proxy_a = FileProxy(console, file)
    proxy_b = FileProxy(console, file)
    proxy_a.write("\x1b[2Aa\n")
    assert file.getvalue() == "a\n"
    with console.use_channel_policy(ChannelPolicy(control=True)):
        # Policy pushed after both proxies were constructed
        proxy_b.write("\x1b[2Ab\n")
        proxy_a.write("\x1b[2Ac\n")
    proxy_b.write("\x1b[2Ad\n")
    assert file.getvalue() == "a\n\x1b[2Ab\n\x1b[2Ac\nd\n"


# --------------------------------------------------------------------------
# capture & record
# --------------------------------------------------------------------------


def test_capture_filters_control_plain_text():
    console = make_console(force_terminal=False)
    proxy = FileProxy(console, io.StringIO())
    with console.capture() as capture:
        with console.use_channel_policy(ChannelPolicy(control=True)):
            proxy.write("a\x1b[2Ab\x07c\n")
    assert capture.get() == "abc\n"


def test_capture_with_terminal_keeps_control_in_buffer():
    console = make_console(force_terminal=True)
    proxy = FileProxy(console, io.StringIO())
    with console.capture() as capture:
        with console.use_channel_policy(ChannelPolicy(control=True)):
            proxy.write("a\x1b[2Ab\n")
    # Current capture logic renders control codes for terminals.
    assert capture.get() == "a\x1b[2Ab\n"


def test_record_records_control_but_exports_filter_it():
    file = io.StringIO()
    console = make_console(
        force_terminal=True, color_system="truecolor", record=True, file=file
    )
    proxy = FileProxy(console, file)
    with console.use_channel_policy(ChannelPolicy(control=True, title=True)):
        proxy.write("\x1b[31mA\x1b[2KB\x07\x1b]0;T\x07\n")

    # Control segments are recorded
    assert any(segment.control for segment in console._record_buffer)

    plain = console.export_text(clear=False)
    assert plain == "AB\n"
    html = console.export_html(clear=False)
    assert "\x1b" not in html
    assert '<span class="r1">A</span>' in html
    assert '<span class="r1">B</span>' in html
    svg = console.export_svg(clear=False)
    assert "\x1b" not in svg
    assert ">A</text>" in svg and ">B</text>" in svg
    # Styles are still recorded
    assert "\x1b[31m" in file.getvalue()


# --------------------------------------------------------------------------
# Live integration
# --------------------------------------------------------------------------


def test_live_inbound_control_above_live_and_not_frozen():
    console = make_console()
    console.begin_capture()
    proxy = FileProxy(console, io.StringIO())
    live_renderable = Text("LIVE\ndisplay", end="")
    with Live(live_renderable, console=console, auto_refresh=False) as live:
        # Policy is not frozen at Live start: control off while Live is active
        proxy.write("off\x1b[5Aline\n")
        with console.use_channel_policy(ChannelPolicy(control=True)):
            proxy.write("foo\x1b[3Dbar\n")
        live.refresh()
    output = console.end_capture()
    # Closed while Live active: dropped
    assert "\x1b[5A" not in output
    # Open while Live active: present, in stream order, exactly once (not merged
    # in to LiveRender's own reposition/erase sequences)
    assert "foo\x1b[3Dbar\n" in output
    assert output.count("\x1b[3D") == 1
    # LiveRender's own cursor sequences are still emitted separately
    assert "\x1b[1A" in output and "\x1b[2K" in output


def test_live_redirect_proxy_uses_same_stack():
    import sys

    console = make_console()
    console.begin_capture()
    with Live(console=console, auto_refresh=False):
        assert isinstance(sys.stdout, FileProxy)
        with console.use_channel_policy(ChannelPolicy(control=True)):
            sys.stdout.write("via-redirect\x1b[4D!\n")
    output = console.end_capture()
    assert "via-redirect\x1b[4D!\n" in output
    assert not isinstance(sys.stdout, FileProxy)


# --------------------------------------------------------------------------
# Other render paths
# --------------------------------------------------------------------------


def test_from_ansi_is_style_only():
    text = Text.from_ansi("\x1b[31mhi\x1b[2K\x1b[1A\x07\x1b]0;T\x07")
    assert text.plain == "hi"
    console = make_console()
    segments = list(console.render(text, console.options))
    assert list(Segment.filter_control(segments, is_control=True)) == []
    rendered = console._render_buffer(segments)
    assert "\x1b[" not in rendered and "\x07" not in rendered


def test_rich_handler_never_emits_control():
    console = make_console(force_terminal=True)
    from rich.logging import RichHandler

    handler = RichHandler(console=console)
    record = logging.LogRecord(
        "name",
        logging.INFO,
        __file__,
        1,
        "message \x1b[2K\x1b[1A\x07",
        None,
        None,
    )
    # Even with control and title wide open on the stack, the logging render
    # path must never emit control/title segments in to the log grid.
    with console.use_channel_policy(
        ChannelPolicy(style=False, control=True, title=True)
    ):
        message = handler.render_message(record, record.getMessage())
        segments = list(console.render(message, console.options))
    assert list(Segment.filter_control(segments, is_control=True)) == []


def test_jupyter_display_path_filters_control_and_title():
    console = Console(
        force_jupyter=True, legacy_windows=False, color_system="truecolor"
    )
    renderable = Group(
        Text("hi", end=""),
        Control((ControlType.CURSOR_UP, 2)),
        Control.title("T"),
    )
    segments = list(console.render(renderable, console.options))
    html = _render_segments(segments)
    text = console._render_buffer(segments)
    assert "hi" in html
    assert "\x1b[" not in html and "\x07" not in html
    assert text == "hi"


def test_syntax_path_unaffected_by_channel_stack():
    from rich.syntax import Syntax

    console = make_console(color_system="truecolor")
    syntax = Syntax("print('hello')\n", "python", line_numbers=False)
    with console.capture() as capture:
        with console.use_channel_policy(
            ChannelPolicy(style=False, control=True, title=True)
        ):
            console.print(syntax)
    assert "print" in capture.get()


def test_console_constructor_takes_no_new_required_args():
    # Constructing with no arguments must still work
    console = Console(legacy_windows=False, _environ={}, file=io.StringIO())
    assert console.channel_policy == DEFAULT_CHANNEL_POLICY
