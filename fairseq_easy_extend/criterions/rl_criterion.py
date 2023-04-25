import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data import Dictionary
from sacrebleu import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from dataclasses import dataclass, field

import math
from fairseq import metrics

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})

@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric

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

        #padding mask, do not remove
        if masks is not None:
            masked_indices = masks.nonzero(as_tuple=True)

            outputs_masked = outputs[masked_indices]
            targets_masked = targets[masked_indices]

        with torch.no_grad():
            logits = F.softmax(outputs_masked, dim=-1)
            sampled_indices = torch.multinomial(logits, 1).squeeze(-1)
            sampled_sentence = sampled_indices.tolist()

            tgt_dict = self.task.target_dictionary
            sampled_sentence_string = tgt_dict.string(sampled_sentence)
            target_sentence = tgt_dict.string(targets_masked.tolist())

            if self.metric == "bleu":
                R = sentence_bleu(target_sentence, [sampled_sentence_string])
                R = R.score  # Convert BLEUScore object to numeric value
            elif self.metric == "meteor":
                R = single_meteor_score(target_sentence, sampled_sentence_string)
            else:
                raise ValueError("Invalid sentence_level_metric. Choose 'bleu' or 'meteor'.")

        log_probs = F.log_softmax(outputs, dim=-1)
        log_probs_selected = log_probs[(*masked_indices, sampled_indices.unsqueeze(-1))].squeeze(-1)
        loss = -log_probs_selected * R
        loss = loss.mean()

        return loss