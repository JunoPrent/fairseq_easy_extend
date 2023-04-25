import torch
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data import Dictionary
from sacrebleu import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from dataclasses import dataclass, field

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="sacrebleu",
                                       metadata={"help": "sentence level metric"})

@register_criterion("rl_loss_new", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.dictionary = task.target_dictionary

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
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        with torch.no_grad():
            # Convert predicted output tokens to sentences
            predicted_sentences = [self.dictionary.string(t) for t in outputs.argmax(dim=-1)]
            # Convert target tokens to sentences
            target_sentences = [self.dictionary.string(t) for t in targets]

            # Calculate sentence-level metric scores
            scores = []
            for pred_sent, tgt_sent in zip(predicted_sentences, target_sentences):
                if self.metric == 'sacrebleu':
                    score = sentence_bleu(pred_sent, [tgt_sent]).score
                elif self.metric == 'meteor':
                    score = single_meteor_score(tgt_sent, pred_sent)
                else:
                    raise ValueError(f"Unsupported sentence-level metric: {self.metric}")
                scores.append(score)
            R = torch.tensor(scores, device=outputs.device)

        # Calculate loss
        log_probs = torch.log_softmax(outputs, dim=-1)
        loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) * R
        loss = loss.mean()

        return loss
