import math
import torch
import torch.nn.functional as F
from argparse import Namespace

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data import Dictionary

from fairseq import utils
from fairseq import metrics
from fairseq.data import encoders

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from dataclasses import dataclass, field



@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})

@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tgt_dict = task.tgt_dict

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss = self._compute_loss(outs, tgt_tokens, masks)

        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def _compute_loss(self, outputs, targets, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)
        # print("outputs: ", outputs)
        # print("targets", targets)

        with torch.no_grad():
            probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
            sample_idx  = torch.multinomial(probs, 1, replacement=True).view(bsz, seq_len)
            # print("sample_idx: ", sample_idx)

            sampled_sentence_string = self.tgt_dict.string(sample_idx) 
            target_sentence = self.tgt_dict.string(targets)
            # print("sampled_sentence_string: ", sampled_sentence_string)

            self.tokenizer = encoders.build_tokenizer(Namespace(tokenizer='moses'))
            sampled_sentence_string = self.tokenizer.decode(sampled_sentence_string)
            target_sentence = self.tokenizer.decode(target_sentence)
            # print("sampled_sentence_string_detokenized: ", sampled_sentence_string)

            if self.metric == "bleu":
                R = sentence_bleu(references=[target_sentence.split()], hypothesis=sampled_sentence_string.split())
            elif self.metric == "meteor":
                R = single_meteor_score(target_sentence, sampled_sentence_string)
            else:
                raise ValueError("Invalid sentence_level_metric. Choose 'bleu' or 'meteor'.")

        # Expand the reward to shape BxT
        print("Reward:", R)
        R = torch.tensor(R).expand((bsz, seq_len)).float().to(targets.device)
        print("Reward shape: ", R.shape)
        
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
            R, sample_idx = R[masks], sample_idx[masks]

        # print("masked outputs: ", outputs)
        log_probs = F.log_softmax(outputs, dim=-1)
        print("log_probs.shape: ", log_probs.shape)
        print("log_probs: ", log_probs)
        log_probs_sampled = torch.gather(log_probs, 1, sample_idx.unsqueeze(2))
        print("log_probs_sampled.shape: ", log_probs_sampled.shape)
        print("log_probs_sampled: ", log_probs_sampled)
        loss = -(log_probs_sampled.squeeze() * R)
        loss = loss.mean()
        print("loss:", loss)
        return loss
        
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )