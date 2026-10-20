import re
import sys
from contextlib import suppress
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

from .color import Color
from .segment import ControlCode, ControlType
from .style import Style
from .text import Text

if TYPE_CHECKING:
    from .channel import ChannelPolicy

re_ansi = re.compile(
    r"""
(?:\x1b[0-?])|
(?:\x1b\](.*?)\x1b\\)|
(?:\x1b([(@-Z\\-_]|\[[0-?]*[ -/]*[@-~]))
""",
    re.VERBOSE,
)


class _AnsiToken(NamedTuple):
    """Result of ansi tokenized string."""

    plain: str = ""
    sgr: Optional[str] = ""
    osc: Optional[str] = ""


def _ansi_tokenize(ansi_text: str) -> Iterable[_AnsiToken]:
    """Tokenize a string in to plain text and ANSI codes.

    Args:
        ansi_text (str): A String containing ANSI codes.

    Yields:
        AnsiToken: A named tuple of (plain, sgr, osc)
    """

    position = 0
    sgr: Optional[str]
    osc: Optional[str]
    for match in re_ansi.finditer(ansi_text):
        start, end = match.span(0)
        osc, sgr = match.groups()
        if start > position:
            yield _AnsiToken(ansi_text[position:start])
        if sgr:
            if sgr == "(":
                position = end + 1
                continue
            if sgr.endswith("m"):
                yield _AnsiToken("", sgr[1:-1], osc)
        else:
            yield _AnsiToken("", sgr, osc)
        position = end
    if position < len(ansi_text):
        yield _AnsiToken(ansi_text[position:])


class _AnsiEventKind(IntEnum):
    """The kind of thing found while scanning inbound ANSI text."""

    TEXT = 1
    SGR = 2
    OSC = 3
    CSI = 4
    ESCAPE = 5
    BELL = 6


class _AnsiEvent(NamedTuple):
    """A single lexical event in an inbound ANSI byte stream.

    ``text`` holds the literal run for :attr:`_AnsiEventKind.TEXT`, the SGR
    parameter string for :attr:`_AnsiEventKind.SGR` and the OSC body for
    :attr:`_AnsiEventKind.OSC`. CSI events carry their parameter bytes,
    intermediate bytes and final byte separately.
    """

    kind: _AnsiEventKind
    text: str = ""
    params: str = ""
    intermediates: str = ""
    final: str = ""


# CSI: ESC [ parameter-bytes intermediate-bytes final-byte (ECMA-48)
_re_csi = re.compile(r"\[([0-9:;<=>?]*)([ -/]*)([@-~])")

# Final bytes of a two byte escape sequence consumed by the legacy tokenizer
# (ESC ( 0x28 .. ESC Z 0x5A, ESC \ 0x5C, ESC _ 0x5F).
_SIMPLE_ESCAPE_MIN, _SIMPLE_ESCAPE_MAX = 0x28, 0x5A


def _is_simple_escape(char: str) -> bool:
    code = ord(char)
    return _SIMPLE_ESCAPE_MIN <= code <= _SIMPLE_ESCAPE_MAX or code in (0x5C, 0x5F)


def _find_osc_terminator(text: str, start: int) -> Optional[Tuple[int, int]]:
    """Find the end of an OSC string beginning at ``start``.

    Returns the index of the first byte of the terminator and its length,
    or None if no terminator is present. Both BEL (``\\x07``) and ST
    (``ESC \\``) are accepted.
    """
    bell_index = text.find("\x07", start)
    string_terminator_index = text.find("\x1b\\", start)
    if bell_index == -1:
        if string_terminator_index == -1:
            return None
        return string_terminator_index, 2
    if string_terminator_index == -1 or bell_index < string_terminator_index:
        return bell_index, 1
    return string_terminator_index, 2


def _ansi_events(line: str) -> Iterable[_AnsiEvent]:
    """Scan a line in to ordered ANSI events.

    Tokenization is independent of any channel policy: every recognized
    construct is reported, and callers decide what to do with it.
    """
    length = len(line)
    index = 0
    text_start = 0

    while index < length:
        char = line[index]
        if char == "\x07":
            if index > text_start:
                yield _AnsiEvent(_AnsiEventKind.TEXT, line[text_start:index])
            yield _AnsiEvent(_AnsiEventKind.BELL)
            index += 1
            text_start = index
        elif char == "\x1b":
            if index + 1 >= length:
                # A dangling ESC is left as plain text.
                index += 1
                continue
            next_char = line[index + 1]
            if next_char == "[":
                match = _re_csi.match(line, index + 1)
                if match is not None:
                    if index > text_start:
                        yield _AnsiEvent(_AnsiEventKind.TEXT, line[text_start:index])
                    params, intermediates, final = match.groups()
                    if final == "m":
                        # Preserve the exact middle of the sequence (parameter
                        # and intermediate bytes) for lenient SGR parsing.
                        yield _AnsiEvent(_AnsiEventKind.SGR, params + intermediates)
                    else:
                        yield _AnsiEvent(
                            _AnsiEventKind.CSI,
                            params=params,
                            intermediates=intermediates,
                            final=final,
                        )
                    index = match.end()
                    text_start = index
                    continue
                # Malformed CSI: leave the ESC as plain text and rescan.
                index += 1
            elif next_char == "]":
                terminator = _find_osc_terminator(line, index + 2)
                if terminator is not None:
                    terminator_start, terminator_length = terminator
                    if index > text_start:
                        yield _AnsiEvent(_AnsiEventKind.TEXT, line[text_start:index])
                    yield _AnsiEvent(
                        _AnsiEventKind.OSC,
                        line[index + 2 : terminator_start],
                    )
                    index = terminator_start + terminator_length
                    text_start = index
                    continue
                # Unterminated OSC: leave the ESC as plain text and rescan.
                index += 1
            elif _is_simple_escape(next_char):
                if index > text_start:
                    yield _AnsiEvent(_AnsiEventKind.TEXT, line[text_start:index])
                yield _AnsiEvent(_AnsiEventKind.ESCAPE)
                # Charset designators (e.g. ESC ( B) consume one final byte.
                index += 3 if next_char == "(" and index + 2 < length else 2
                text_start = index
                continue
            else:
                index += 1
        else:
            index += 1

    if text_start < length:
        yield _AnsiEvent(_AnsiEventKind.TEXT, line[text_start:])


# CSI final bytes which map directly on to existing ControlType members, with
# the parameter to use when the sequence carries none.
_CSI_CONTROL: Dict[str, Tuple[ControlType, int]] = {
    "A": (ControlType.CURSOR_UP, 1),
    "B": (ControlType.CURSOR_DOWN, 1),
    "C": (ControlType.CURSOR_FORWARD, 1),
    "D": (ControlType.CURSOR_BACKWARD, 1),
    "G": (ControlType.CURSOR_MOVE_TO_COLUMN, 1),
    "K": (ControlType.ERASE_IN_LINE, 0),
}


def _first_csi_parameter(params: str, default: int) -> Optional[int]:
    """Parse the first numeric CSI parameter, leniently."""
    first = params.split(";", 1)[0].split(":", 1)[0]
    if first == "":
        return default
    if not first.isdigit():
        # Private-mode markers (e.g. "?25") or otherwise non-numeric
        # parameters do not map on to any ControlType.
        return None
    return int(first)


def csi_to_control(event: _AnsiEvent) -> Optional[ControlCode]:
    """Map a CSI event on to a control code, or return None if it can't map.

    Only parameterized CSI sequences without private markers or intermediate
    bytes are considered.
    """
    if event.kind is not _AnsiEventKind.CSI or event.intermediates:
        return None
    mapping = _CSI_CONTROL.get(event.final)
    if mapping is None:
        return None
    control_type, default = mapping
    parameter = _first_csi_parameter(event.params, default)
    if parameter is None:
        return None
    if control_type == ControlType.CURSOR_MOVE_TO_COLUMN:
        # ANSI columns are 1 based; Rich's parameter is 0 based.
        parameter = max(0, parameter - 1)
    return (control_type, parameter)


def osc_to_title(body: str) -> Optional[str]:
    """Extract a window title from an OSC body, or return None.

    OSC 0 sets the icon name and window title; OSC 2 sets the window title.
    """
    if body.startswith(("0;", "2;")):
        return body[2:]
    return None


SGR_STYLE_MAP = {
    1: "bold",
    2: "dim",
    3: "italic",
    4: "underline",
    5: "blink",
    6: "blink2",
    7: "reverse",
    8: "conceal",
    9: "strike",
    21: "underline2",
    22: "not dim not bold",
    23: "not italic",
    24: "not underline",
    25: "not blink",
    26: "not blink2",
    27: "not reverse",
    28: "not conceal",
    29: "not strike",
    30: "color(0)",
    31: "color(1)",
    32: "color(2)",
    33: "color(3)",
    34: "color(4)",
    35: "color(5)",
    36: "color(6)",
    37: "color(7)",
    39: "default",
    40: "on color(0)",
    41: "on color(1)",
    42: "on color(2)",
    43: "on color(3)",
    44: "on color(4)",
    45: "on color(5)",
    46: "on color(6)",
    47: "on color(7)",
    49: "on default",
    51: "frame",
    52: "encircle",
    53: "overline",
    54: "not frame not encircle",
    55: "not overline",
    90: "color(8)",
    91: "color(9)",
    92: "color(10)",
    93: "color(11)",
    94: "color(12)",
    95: "color(13)",
    96: "color(14)",
    97: "color(15)",
    100: "on color(8)",
    101: "on color(9)",
    102: "on color(10)",
    103: "on color(11)",
    104: "on color(12)",
    105: "on color(13)",
    106: "on color(14)",
    107: "on color(15)",
}


class AnsiDecoder:
    """Translate ANSI code in to styled Text.

    This is a style-only decoder: SGR and OSC hyperlinks are captured, while
    control sequences and window titles are ignored. It backs
    :meth:`Text.from_ansi` and therefore paths such as log message decoding.
    """

    def __init__(self) -> None:
        self.style = Style.null()

    def decode(self, terminal_text: str) -> Iterable[Text]:
        """Decode ANSI codes in an iterable of lines.

        Args:
            terminal_text: Output potentially containing ANSI escape sequences.

        Yields:
            Text: Marked up Text.
        """
        for line in re.split(r"(?<=\n)", terminal_text):
            yield self.decode_line(line.rstrip("\n"))

    def _apply_sgr(self, sgr: str) -> None:
        """Apply the parameter string of an SGR (``ESC [ ... m``) sequence."""
        from_ansi = Color.from_ansi
        from_rgb = Color.from_rgb
        _Style = Style
        # Translate in to semi-colon separated codes
        # Ignore invalid codes, because we want to be lenient
        codes = [
            min(255, int(_code) if _code else 0)
            for _code in sgr.split(";")
            if _code.isdigit() or _code == ""
        ]
        iter_codes = iter(codes)
        for code in iter_codes:
            if code == 0:
                # reset
                self.style = _Style.null()
            elif code in SGR_STYLE_MAP:
                # styles
                self.style += _Style.parse(SGR_STYLE_MAP[code])
            elif code == 38:
                #  Foreground
                with suppress(StopIteration):
                    color_type = next(iter_codes)
                    if color_type == 5:
                        self.style += _Style.from_color(
                            from_ansi(next(iter_codes))
                        )
                    elif color_type == 2:
                        self.style += _Style.from_color(
                            from_rgb(
                                next(iter_codes),
                                next(iter_codes),
                                next(iter_codes),
                            )
                        )
            elif code == 48:
                # Background
                with suppress(StopIteration):
                    color_type = next(iter_codes)
                    if color_type == 5:
                        self.style += _Style.from_color(
                            None, from_ansi(next(iter_codes))
                        )
                    elif color_type == 2:
                        self.style += _Style.from_color(
                            None,
                            from_rgb(
                                next(iter_codes),
                                next(iter_codes),
                                next(iter_codes),
                            ),
                        )

    def _apply_hyperlink(self, osc: str) -> None:
        """Apply an OSC 8 hyperlink body to the current style state."""
        _params, semicolon, link = osc[2:].partition(";")
        if semicolon:
            self.style = self.style.update_link(link or None)

    def decode_line(self, line: str) -> Text:
        """Decode a line containing ansi codes.

        Args:
            line (str): A line of terminal output.

        Returns:
            Text: A Text instance marked up according to ansi codes.
        """
        text = Text()
        append = text.append
        line = line.rsplit("\r", 1)[-1]
        for event in _ansi_events(line):
            if event.kind is _AnsiEventKind.TEXT:
                append(event.text, self.style or None)
            elif event.kind is _AnsiEventKind.SGR:
                self._apply_sgr(event.text)
            elif event.kind is _AnsiEventKind.OSC and event.text.startswith("8;"):
                self._apply_hyperlink(event.text)
        return text


#: A decoded item: ``("text", Text)``, ``("control", ControlCode)`` or
#: ``("title", str)``.
AnsiChannelItem = NamedTuple(
    "AnsiChannelItem",
    [("kind", str), ("value", Any)],
)


class AnsiChannelDecoder:
    """Decode inbound ANSI text under a :class:`~rich.channel.ChannelPolicy`.

    Unlike :class:`AnsiDecoder`, this splits the byte stream in to three
    exits: styled :class:`~rich.text.Text`, control codes and window titles,
    in stream order. The style state is shared with an ``AnsiDecoder`` and is
    only advanced when the style channel is open, so sequences received while
    the channel is closed leave no residual state.
    """

    def __init__(self) -> None:
        self._style_decoder = AnsiDecoder()

    def decode_line(self, line: str, policy: "ChannelPolicy") -> List[AnsiChannelItem]:
        """Decode a single line into an ordered list of channel items.

        Args:
            line (str): A line of terminal output.
            policy (ChannelPolicy): The policy in effect while decoding.

        Returns:
            List[AnsiChannelItem]: Text, control and title items in byte
                stream order.
        """
        items: List[AnsiChannelItem] = []
        # Chunks carry no end of their own: the caller adds line endings once,
        # after the full stream (including any trailing control sequences).
        text = Text(end="")

        def flush_text() -> None:
            if text.plain:
                items.append(AnsiChannelItem("text", text))

        line = line.rsplit("\r", 1)[-1]

        def new_text() -> Text:
            return Text(end="")

        for event in _ansi_events(line):
            kind = event.kind
            if kind is _AnsiEventKind.TEXT:
                text.append(
                    event.text,
                    (self._style_decoder.style or None) if policy.style else None,
                )
            elif kind is _AnsiEventKind.SGR:
                if policy.style:
                    self._style_decoder._apply_sgr(event.text)
            elif kind is _AnsiEventKind.OSC:
                body = event.text
                if body.startswith("8;"):
                    if policy.style:
                        self._style_decoder._apply_hyperlink(body)
                elif policy.title:
                    title = osc_to_title(body)
                    if title is not None:
                        flush_text()
                        text = new_text()
                        items.append(AnsiChannelItem("title", title))
            elif kind is _AnsiEventKind.CSI:
                if policy.control:
                    control_code = csi_to_control(event)
                    if control_code is not None:
                        flush_text()
                        text = new_text()
                        items.append(AnsiChannelItem("control", control_code))
            elif kind is _AnsiEventKind.BELL:
                if policy.control:
                    flush_text()
                    text = new_text()
                    items.append(AnsiChannelItem("control", (ControlType.BELL,)))
            # ESCAPE events (charset designators etc.) are always discarded.

        flush_text()
        return items


if sys.platform != "win32" and __name__ == "__main__":  # pragma: no cover
    import io
    import os
    import pty
    import sys

    decoder = AnsiDecoder()

    stdout = io.BytesIO()

    def read(fd: int) -> bytes:
        data = os.read(fd, 1024)
        stdout.write(data)
        return data

    pty.spawn(sys.argv[1:], read)

    from .console import Console

    console = Console(record=True)

    stdout_result = stdout.getvalue().decode("utf-8")
    print(stdout_result)

    for line in decoder.decode(stdout_result):
        console.print(line)

    console.save_html("stdout.html")
