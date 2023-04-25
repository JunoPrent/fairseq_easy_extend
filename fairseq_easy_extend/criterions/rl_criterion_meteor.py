import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import meteor

from dataclasses import dataclass, field

@dataclass
class RLCriterionMeteorConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="meteor",
                                       metadata={"help": "sentence level metric"})

@register_criterion("rl_criterion_meteor", dataclass=RLCriterionMeteorConfig)
class RLCriterionMeteor(FairseqCriterion):
    def init(self, task, sentence_level_metric):
        super().init(task)
        self.metric = sentence_level_metric

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

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
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        # Convert sampled and target sentences to strings
        with torch.no_grad():
            sampled_sentence = self.task.target_dictionary.string(outputs.argmax(-1).cpu().numpy())
            target_sentence = self.task.target_dictionary.string(targets.cpu().numpy())

            # Calculate the METEOR score
            meteor_score = meteor([target_sentence], sampled_sentence)

        # Compute the loss
        log_prob = torch.log_softmax(outputs, dim=-1)
        log_prob_sampled = torch.gather(log_prob, 2, outputs.argmax(-1).unsqueeze(-1)).squeeze(-1)
        loss = -(log_prob_sampled * meteor_score).mean()

        return loss