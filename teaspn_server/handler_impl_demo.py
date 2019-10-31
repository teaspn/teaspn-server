import json
import logging
import re
import time
from multiprocessing import Process, Value, JoinableQueue, Queue
from typing import Dict, List, Optional

import language_check
import neuralcoref
import requests
import spacy
from nltk.corpus import wordnet
from overrides import overrides
from retrying import retry

from teaspn_server.protocol import (
    CodeAction, Command, CompletionItem, CompletionList, Diagnostic, DiagnosticSeverity, Example,
    Hover, Location, Position, Range, SyntaxHighlight, TextEdit, WorkspaceEdit
)
from teaspn_server.teaspn_handler import TeaspnHandler
from teaspn_server.gpt2completion_handler import GPT2CompletionHandler
from teaspn_server.paraphrase_handler import ParaphraseHandler

logger = logging.getLogger(__name__)


class ColorSetting:

    dep2color = {'nsubj': 'salmon',
                 'ROOT': 'green',
                 'dobj': 'skyblue'}


class ExampleHandler(object):

    def __init__(self):
        self._requests = JoinableQueue()
        self._results = Queue()
        self._busy = Value('i', 0)

    @retry(stop_max_attempt_number=5)
    def search_tatoeba_examples_several_attempts(self, query):
        response = requests.post(
            url=('http://elasticsearch:9200/tatoeba_ja/_search'),
            data=json.dumps({'size': 30,
                             'query': {'match_phrase': {'key': query}}}),
            headers={'Content-Type': 'application/json'},
        )
        if response.status_code == 200:
            response_data = response.json()
            logger.info('Received response: %r', response_data)
            for text in response_data['hits']['hits']:
                self._results.put(
                    (text['_source']['value'], text['_source']['key']))
            return True
        else:
            raise IOError("connection fails")
            time.sleep(1.)

    def loop(self):
        while True:
            if not self._requests.empty():
                self._busy.value = 1
                query = self._requests.get()
                # process

                logger.info('IN Example Searching Thread: target_text=%s', query)

                self.search_tatoeba_examples_several_attempts(query)
                self._requests.task_done()
                self._busy.value = 0

            time.sleep(1.)

    def make_request(self, text):
        self._requests.put(text)
        self._requests.join()

        results = []
        while not self._results.empty():
            text = self._results.get()
            results.append(text)

        return results


class TeaspnHandlerImplDemo(TeaspnHandler):
    """
    This is an implementation of ``TeaspnHandler`` for demo purposes.
    """

    def __init__(self):
        super(TeaspnHandler, self).__init__()

        self._detect_ge = language_check.LanguageTool('en-US')
        # Manages a mapping from Diagnostic to replacements obtained from LanguageTool.
        # TODO: maybe move this to the base class?
        self._diag_to_replacements: Dict[Diagnostic, List[str]] = {}

        self._paraphrase_handler = ParaphraseHandler(model_path='model/paraphrase/checkpoint_best.pt',
                                                     data_path='model/paraphrase',
                                                     tokenizer_path='model/paraphrase/spm/para_nmt.model')

        self._completion_handler = GPT2CompletionHandler(num_samples=4, length=5)

        self._example_handler = ExampleHandler()
        self._example_process = Process(target=self._example_handler.loop)
        self._example_process.start()

        self._nlp = spacy.load('en')

        neuralcoref.add_to_pipe(self._nlp)

        self._freq_words = [word.strip() for word in open('teaspn_server/word.txt')]

    @overrides
    def highlight_syntax(self) -> List[SyntaxHighlight]:
        highlights = []
        for line_id, line_text in enumerate(self._text.splitlines()):

            for tok_id, tok in enumerate(self._nlp(line_text)):
                if tok.dep_ in ColorSetting.dep2color:
                    rng = Range(start=Position(line=line_id, character=tok.idx),
                                end=Position(line=line_id, character=tok.idx + len(tok.text)))
                    highlights.append(SyntaxHighlight(range=rng,
                                                      type=ColorSetting.dep2color[tok.dep_],
                                                      hoverMessage='dep: {}'.format(tok.dep_)))

        return highlights

    @overrides
    def get_diagnostics(self) -> List[Diagnostic]:
        diagnostics = []
        for line_id, line_text in enumerate(self._text.splitlines()):
            for m in self._detect_ge.check(line_text):
                rng = Range(start=Position(line=line_id, character=m.fromx),
                            end=Position(line=line_id, character=m.tox))
                diagnostic = Diagnostic(range=rng,
                                        severity=DiagnosticSeverity.Error,
                                        message=m.msg)
                diagnostics.append(diagnostic)
                self._diag_to_replacements[diagnostic] = m.replacements
        return diagnostics

    @overrides
    def run_quick_fix(self, range: Range, diagnostics: List[Diagnostic]) -> List[CodeAction]:
        actions = []
        for diag in diagnostics:
            for repl in self._diag_to_replacements[diag]:
                edit = WorkspaceEdit({self._uri: [TextEdit(range=diag.range, newText=repl)]})
                command = Command(title='Quick fix: {}'.format(repl),
                                  command='refactor.rewrite',
                                  arguments=[edit])
                actions.append(CodeAction(title='Quick fix: {}'.format(repl),
                                          kind='quickfix',
                                          command=command))
        return actions

    @overrides
    def run_code_action(self, range: Range) -> List[Command]:
        target_text = self._get_text(range)
        if not target_text:
            return []
        texts = []
        for text in self._paraphrase_handler.generate(target_text):
            texts.append(text)
        commands = []
        for text in texts:
            edit = WorkspaceEdit({self._uri: [TextEdit(range=range, newText=text)]})
            command = Command(title='Suggestion: {}'.format(text),
                              command='refactor.rewrite',
                              arguments=[edit])
            commands.append(command)
        return commands

    @overrides
    def search_example(self, query: str) -> List[Example]:
        examples = self._example_handler.make_request(query)
        example_list = []
        for label, description in examples:
            example = Example(label=label, description=description)
            example_list.append(example)
        return example_list

    @overrides
    def search_definition(self, position: Position, uri: str) -> List[Location]:
        doc = self._nlp(self._text)
        offset = self._position_to_offset(position)
        locations = []
        for coreference in doc._.coref_clusters:
            for mention in coreference.mentions:
                if mention.start_char <= offset <= mention.end_char:
                    rng = Range(start=self._offset_to_position(coreference.main.start_char),
                                end=self._offset_to_position(coreference.main.end_char))
                    locations.append(Location(uri=uri, range=rng))

        return locations

    @overrides
    def get_completion_list(self, position: Position) -> CompletionList:
        offset = self._position_to_offset(position)
        context = self._text[:offset]
        logger.info('Completion: context=%r', context)

        if not context:
            return CompletionList(isIncomplete=False, items=[])

        items = []
        if context.endswith(' '):
            for text in self._completion_handler.generate(context):
                if re.match(r'[.,:;]', text):
                    position_dict = position.to_dict()
                    position_dict['character'] += -1
                    position = Position.from_dict(position_dict)

                rng = Range(start=position, end=position)
                items.append(CompletionItem(label=text,
                                            textEdit=TextEdit(range=rng, newText=text)))

        else:
            query = context.split()[-1]
            texts = [word for word in self._freq_words if word.startswith(query)]
            for text in texts:
                rng = Range(start=position, end=position)
                new_text = text.replace(context.split()[-1], '')
                text_edit = TextEdit(range=rng, newText=new_text)
                items.append(CompletionItem(label=text, textEdit=text_edit))

        return CompletionList(isIncomplete=False, items=items)

    @overrides
    def hover(self, position: Position) -> Optional[Hover]:
        # NOTE: currently this implements a very simple PoC where the word at position
        # is matched against WordNet synsets and the first result is returned, regardless of POS
        word = self._get_word_at(position)
        if not word:
            return None

        synsets = wordnet.synsets(word)
        if not synsets:
            return None

        pos, definition = synsets[0].pos(), synsets[0].definition()
        return Hover(contents=f'{pos}: {definition}')
