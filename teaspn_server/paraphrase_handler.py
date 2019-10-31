import os
import logging
from collections import namedtuple
from contextlib import redirect_stdout

import torch
import sentencepiece as spm
from fairseq import options, tasks, utils, checkpoint_utils


logger = logging.getLogger(__name__)

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')


class ParaphraseHandler(object):
    def __init__(self, model_path, data_path, tokenizer_path, beam=3, n_best=3, diverse_beam_group=3,
                 diverse_beam_strength=0.5, source_lang='source.spm', target_lang='target.spm'):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        self.parser = options.get_generation_parser()
        self.args = options.parse_args_and_arch(self.parser,
                                                input_args=['--cpu',
                                                            f'--path={model_path}',
                                                            f'--beam={beam}',
                                                            f'--nbest={n_best}',
                                                            f'--diverse-beam-groups={diverse_beam_group}',
                                                            f'--diverse-beam-strength={diverse_beam_strength}',
                                                            f'--source-lang={source_lang}',
                                                            f'--target-lang={target_lang}',
                                                            f'--task=translation',
                                                            '--remove-bpe=sentencepiece',
                                                            data_path])

        # self.task = TranslationTask.setup_task(self.args)
        with redirect_stdout(open(os.devnull, 'w')):
            self.task = tasks.setup_task(self.args)
        self.models, _ = checkpoint_utils.load_model_ensemble(self.args.path.split(':'), task=self.task,
                                                              arg_overrides=eval(self.args.model_overrides))
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                                        need_attn=self.args.print_alignment)

        self.generator = self.task.build_generator(self.args)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self._max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )
        self.use_cuda = False

    def generate(self, text_input: str) -> list:

        text_input = ' '.join(self.tokenizer.EncodeAsPieces(text_input))
        outputs = []
        start_id = 0

        results = []
        for batch in self._make_batches([text_input]):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

            # sort output to match input order
        for _, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                outputs.append(hypo_str)
            # update running id counter
        return list(set(outputs))

    def _make_batches(self, lines):
        tokens = [
            self.task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
            for src_str in lines
        ]
        lengths = torch.LongTensor([t.numel() for t in tokens])
        itr = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self._max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            )
