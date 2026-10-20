from typing import List, NamedTuple


class ChannelPolicy(NamedTuple):
    """Policy for the three inbound ANSI channels.

    Unlike the theme stack (which only maps style names), this controls which
    kinds of ANSI sequences found in a redirected byte stream (see
    :class:`~rich.file_proxy.FileProxy`) are accepted at all.

    Attributes:
        style: When True, SGR sequences and OSC hyperlinks (OSC 8) are decoded
            in to styled :class:`~rich.text.Text`. When False, they must not
            advance decoder style state and the text is written plain.
        control: When True, CSI sequences that map on to an existing
            :class:`~rich.segment.ControlType` (inline erase, cursor movement
            and the bell) are emitted via :meth:`~rich.console.Console.control`.
            When False, they are discarded rather than written as characters.
        title: When True, OSC window title sequences (OSC 0 / OSC 2) are
            emitted as window title controls. When False, they are discarded.
    """

    style: bool = True
    control: bool = False
    title: bool = False


#: Policy used when no policy has been pushed. Preserves the historical
#: FileProxy behaviour: styled text only, everything else dropped.
DEFAULT_CHANNEL_POLICY: ChannelPolicy = ChannelPolicy()


class ChannelStackError(Exception):
    """Raised for errors related to the channel policy stack."""


class ChannelStack:
    """A stack of :class:`ChannelPolicy` instances.

    The stack is deliberately independent of the
    :class:`~rich.theme.ThemeStack`: the two may be pushed and popped in
    interleaved order and each pops in its own order. Reading an empty stack
    yields :data:`DEFAULT_CHANNEL_POLICY`.
    """

    def __init__(self) -> None:
        self._entries: List[ChannelPolicy] = []

    @property
    def top(self) -> ChannelPolicy:
        """The policy currently in effect, or the default if none is pushed."""
        return self._entries[-1] if self._entries else DEFAULT_CHANNEL_POLICY

    def push_policy(self, policy: ChannelPolicy) -> None:
        """Push a policy on to the top of the stack.

        Args:
            policy (ChannelPolicy): A channel policy instance.
        """
        self._entries.append(policy)

    def pop_policy(self) -> None:
        """Pop (and discard) the top-most policy."""
        if not self._entries:
            raise ChannelStackError("Unable to pop channel policy from empty stack")
        self._entries.pop()
