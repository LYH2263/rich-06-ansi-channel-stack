import io
from typing import IO, TYPE_CHECKING, Any, List

from .ansi import AnsiChannelDecoder
from .console import Group, NewLine
from .control import Control
from .text import Text

if TYPE_CHECKING:
    from .console import Console, RenderableType

#: Separator inserted between complete lines, without an end of its own.
_LINE_SEPARATOR = Text("\n", end="")


class FileProxy(io.TextIOBase):
    """Wraps a file (e.g. sys.stdout) and redirects writes to a console."""

    def __init__(self, console: "Console", file: IO[str]) -> None:
        self.__console = console
        self.__file = file
        self.__buffer: List[str] = []
        self.__ansi_decoder = AnsiChannelDecoder()

    @property
    def rich_proxied_file(self) -> IO[str]:
        """Get proxied file."""
        return self.__file

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__file, name)

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        buffer = self.__buffer
        lines: List[str] = []
        while text:
            line, new_line, text = text.partition("\n")
            if new_line:
                lines.append("".join(buffer) + line)
                buffer.clear()
            else:
                buffer.append(line)
                break
        if lines:
            console = self.__console
            # Read the stack top for every write; it must not be frozen when a
            # Live display (which installs this proxy) starts.
            policy = console.channel_policy
            renderables: List["RenderableType"] = []
            append = renderables.append
            for line_index, line in enumerate(lines):
                if line_index:
                    append(_LINE_SEPARATOR)
                # Decode in to the three exits, keeping byte stream order:
                # styled Text is printed, control codes and window titles are
                # emitted through console.control (see the Group below).
                for item in self.__ansi_decoder.decode_line(line, policy):
                    if item.kind == "text":
                        append(item.value)
                    elif item.kind == "title":
                        append(Control.title(item.value))
                    else:
                        append(Control(item.value))
            # A single line ending for the whole batch, emitted last so that
            # trailing control sequences stay on the line they arrived on.
            append(NewLine())
            with console:
                # The decoded Text is printed as a renderable: no markup
                # parsing and no highlighter are applied, and one print call
                # means Live wraps the whole interleaved stream (rather than
                # merging controls in to its own cursor sequences).
                console.print(Group(*renderables))
        return len(text)

    def flush(self) -> None:
        output = "".join(self.__buffer)
        if output:
            self.__console.print(output)
        del self.__buffer[:]

    def fileno(self) -> int:
        return self.__file.fileno()

    def isatty(self) -> bool:
        return self.__file.isatty()
