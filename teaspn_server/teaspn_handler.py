import logging
import re
from typing import List, Optional

from teaspn_server.protocol import (
    CodeAction, Command, CompletionList, Diagnostic, Hover, Location,
    Position, Range, SyntaxHighlight
)


logger = logging.getLogger(__name__)


class TeaspnHandler(object):
    """
    This is the abstract base class for handling requests (method calls) from ``TeaspnServer``.
    Language smartness developers are expected to inherit this class and create their own
    implementation of this class.

    This class also provides some low-level utility methods for keeping the document in sync with
    the client side.
    """

    def __init__(self):
        self._line_offsets = [0]
        self._uri = None
        self._text = ""

    def _position_to_offset(self, position: Position) -> int:
        return self._line_offsets[position.line] + position.character

    def _offset_to_position(self, offset: int) -> Position:
        line = 0
        for i, n in enumerate(self._line_offsets):
            if n > offset:
                line = i - 1
                break

        character = offset - line
        return Position(line=line, character=character)

    def _recompute_line_offsets(self):
        self._line_offsets = [0] + [m.start() + 1 for m in re.finditer('\n', self._text)]
        # NOTE: \r\n may also need to be considered.

    def _get_text(self, range: Range) -> str:
        start_offset = self._position_to_offset(range.start)
        end_offset = self._position_to_offset(range.end)
        return self._text[start_offset:end_offset]

    def _get_line(self, line: int) -> str:
        """
        Returns the text on the specified line.
        """

        line_offsets_with_sentinel = self._line_offsets + [len(self._text)]
        return self._text[line_offsets_with_sentinel[line]:line_offsets_with_sentinel[line+1]]

    def _get_word_at(self, position: Position) -> Optional[str]:
        """
        Returns the word at position.

        This method gets the text for the cursor line, and then
        finds the word that encompasses the cursor. This is a bit inefficient, but simple.
        (otherwise you'd need to scan the string in both directions and find punctuations etc.,
        which is a lot more complicated).
        """

        line = self._get_line(position.line)

        for match in re.finditer(r'\w+', line):
            if match.start() <= position.character <= match.end():
                return match.group(0)

        return None

    def initialize_document(self, uri: str, text: str):
        logger.debug('Initialized document: uri=%s, text=%s', uri, text)
        self._uri = uri
        self._text = text
        self._recompute_line_offsets()

    def update_document(self, range: Range, text: str):
        start_offset = self._position_to_offset(range.start)
        end_offset = self._position_to_offset(range.end)
        self._text = self._text[:start_offset] + text + self._text[end_offset:]
        # self._text = text
        self._recompute_line_offsets()
        logger.debug('Updated document: text=%s', self._text)

    def highlight_syntax(self) -> List[SyntaxHighlight]:
        raise NotImplementedError

    def get_diagnostics(self) -> List[Diagnostic]:
        raise NotImplementedError

    def run_quick_fix(self, range: Range, diagnostics: List[Diagnostic]) -> List[CodeAction]:
        raise NotImplementedError

    def run_code_action(self, range: Range) -> List[Command]:
        raise NotImplementedError

    def search_example(self, query: str) -> List:
        # TODO: fix the protocol definition of search example
        raise NotImplementedError

    def search_definition(self, position: Position, uri: str) -> List[Location]:
        raise NotImplementedError

    def get_completion_list(self, position: Position) -> CompletionList:
        raise NotImplementedError

    def hover(self, position: Position) -> Optional[Hover]:
        raise NotImplementedError
