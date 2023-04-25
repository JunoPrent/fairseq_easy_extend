import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring.meteor import MeteorScorer

from dataclasses import dataclass, field

@dataclass
class RLCriterionMeteorConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="meteor",
                                       metadata={"help": "sentence level metric"})

@register_criterion("rl_criterion_meteor", dataclass=RLCriterionMeteorConfig)
class RLCriterionMeteor(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.scorer = MeteorScorer()

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
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        # Convert model outputs to probabilities
        probs = F.softmax(outputs, dim=-1)

        # Sample sentences from the output probabilities
        sampled_sentences = torch.multinomial(probs, 1).squeeze(-1)

        loss = 0
        n_sentences = targets.size(0)
        for i in range(n_sentences):
            # Convert sampled and target sentences to strings
            sampled_sentence = self.task.target_dictionary.string(sampled_sentences[i].unsqueeze(0))
            target_sentence = self.task.target_dictionary.string(targets[i].unsqueeze(0))

            # Calculate the METEOR score
            meteor_score = self.scorer.score(target_sentence, sampled_sentence)

            # Calculate the loss for the current sentence pair
            log_prob = F.log_softmax(outputs[i], dim=-1)
            log_prob_sentence = log_prob[range(len(sampled_sentences[i])), sampled_sentences[i]].sum()
            loss += -log_prob_sentence * meteor_score

        # Calculate the average loss
        loss /= n_sentences

        return loss
