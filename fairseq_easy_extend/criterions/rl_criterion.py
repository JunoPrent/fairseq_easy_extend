
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data import Dictionary
from fairseq.scoring import meteor, bertscore

from dataclasses import dataclass, field

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.task = task
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

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output


    def eval_metric(self, sampled_sentence, target_sentence, method_type='meteor'):
        """
        Compute the evaluation metric between the predicted and target sentences.

        Args:
            sampled_sentence (str): The predicted sentence as a string.
            target_sentence (str): The target sentence as a string.
            method_type (str): The evaluation metric to use. Defaults to 'meteor'.

        Returns:
            float: The evaluation metric score between the two sentences.
        """

        if method_type.lower() == 'meteor':
            scorer = meteor.MeteorScorer(meteor.MeteorScorerConfig)
            scorer.add_string(target_sentence, sampled_sentence)
            score = scorer.score()
        elif method_type.lower() == 'bertscore':
            scorer = bertscore.BertScoreScorer(bertscore.BertScoreScorerConfig)
            scorer.add_string(target_sentence, sampled_sentence)
            score = scorer.score()
        else:
            raise NotImplementedError(f"Method '{method_type}' not implemented.")

        return score

    def _compute_loss(self, outputs, targets, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)

        probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
        sample_idx  = torch.multinomial(probs, 1, replacement=True).view(bsz, seq_len)

        self.tgt_dict = self.task.tgt_dict
        sampled_sentence_string = self.tgt_dict.string(sample_idx) #here you might also want to remove tokenization and bpe
        target_sentence_string = self.tgt_dict.string(targets)

        with torch.no_grad():
            # R(*) is a number, BLEU, сhrf, etc.
            reward = self.eval_metric(sampled_sentence_string, target_sentence_string, method_type='meteor')
            # Expand it to make it of a shape BxT - each token gets the same reward value
            # (e.g. bleu is 20, so each token gets reward of 20 [20,20,20,20,20])
            reward = torch.Tensor([[reward] * int(seq_len)] * int(bsz))

        # Padding mask, do not remove
        if masks is not None:
            print(outputs.shape, type(outputs), targets.shape, type(targets), sample_idx.shape, type(sample_idx))
            outputs, targets = outputs[masks], targets[masks]
            reward = reward[masks]
            sample_idx = sample_idx[masks]

        # # We take a softmax over outputs
        # softmax_outputs = torch.softmax(outputs, dim=-1)

        # # Argmax over the softmax or sampling (e.g. multinomial)
        # sampled_sentence = torch.argmax(softmax_outputs, dim=-1).tolist()

        # # Convert token indices to string representation using the target dictionary
        # tgt_dict = Dictionary.load("path/to/your/target/dictionary")
        # sampled_sentence_string = tgt_dict.string(sampled_sentence)

        # # Target sentence in string format
        # target_sentence = tgt_dict.string(targets.tolist())


        # Loss = -log_prob(sample_outputs) * R()
        log_probs = F.log_softmax(outputs, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss = -log_probs * reward
        loss = loss.mean()

        return loss