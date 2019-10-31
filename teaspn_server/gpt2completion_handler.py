import time
import logging
from multiprocessing import Value, JoinableQueue, Queue

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logger = logging.getLogger(__name__)


class GPT2CompletionHandler(object):
    def __init__(self, model_path='distilgpt2', tokenizer_path='distilgpt2', length=10, device='cpu', num_samples=1,
                 temperature=1):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.num_samples = num_samples

        self.temperature = temperature
        self.device = device

        self.model.to(self.device)
        self.model.eval()
        self.length = length
        self.past = None
        self.past_context = ""

        if self.length == -1:
            self.length = self.model.config.n_ctx // 2
        elif self.length > self.model.config.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.model.config.n_ctx)

    def generate(self, context):
        context = '<|endoftext|> ' + context
        context_tokens = self.tokenizer.encode(context)
        out = self.sample_sequence(context=context_tokens, )
        out = out[:, len(context_tokens):].tolist()
        return [self.tokenizer.decode(out[i], clean_up_tokenization_spaces=True).strip() for i in
                range(self.num_samples)]

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability > top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: 
            https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            and https://github.com/huggingface/transformers/pull/1333/files#diff-69e141e24d872d0cad3270be7db159e5L104
        """

        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices,
                                                                 src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self, context, top_k=0, top_p=0.0, sample=True):
        """
        Args:
            context:
            top_k:
            top_p:
            sample:

        Note:
             We created based on https://github.com/huggingface/transformers/blob/v2.0.0/examples/run_generation.py
        """

        if context[:len(self.past_context)] == context:
            self.past_context = context
            context = context[len(self.past_context):]
            assert self.past_context != context
            local_past = self.past
        else:
            local_past = None

        context = torch.tensor(context, device=self.device, dtype=torch.long).unsqueeze(0).repeat(self.num_samples, 1)

        logger.info('IN sample_sequence: context=%s', context)
        prev = context
        output = context

        with torch.no_grad():
            for i in range(self.length):
                next_token_logits, local_past = self.model(prev, past=local_past)
                if i == 0:
                    self.past = local_past
                next_token_logits = next_token_logits[:, -1, :] / self.temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                next_token_log_probs = F.softmax(filtered_logits, dim=-1)
                if sample:
                    prev = torch.multinomial(next_token_log_probs, num_samples=1)
                else:
                    _, prev = torch.topk(next_token_log_probs, k=1, dim=-1)
                output = torch.cat((output, prev), dim=1)
        return output
