from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
import math
import torch
import torch.nn as nn

from fairseq import metrics, utils

def cost(x, y):
    len1 = x.size(-2)
    len2 = y.size(-2)
    dim = x.size(-1)
    bsz = x.size(0)
    tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
    ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)

    # cosine
    #f_simi = torch.nn.CosineSimilarity(dim=-1)
    #res = 1. - f_simi(tx, ty)

    # L2
    res = torch.linalg.norm(tx - ty, dim=-1)
    return res


def compute_op_distance_min(x, y, x_mask, y_mask):
    C = cost(x, y)
    # approximate solution
    C.masked_fill_(x_mask.unsqueeze(dim=-1), 0).masked_fill_(y_mask.unsqueeze(dim=-2), 0)
    weight = torch.linalg.norm(x, dim=-1) / torch.linalg.norm(x, dim=-1).sum(dim=-1, keepdim=True)
    res = (C.min(dim=-1)[0] * weight.detach().clone()).sum()
    #res = C.min(dim=-1)[0].mean(dim=-1).sum()
    return res

def compute_kd_loss(probs1, probs2, lprobs1, lprobs2, target, ignore_index, language_tag_id=None):
    if target.dim() == lprobs1.dim() - 1:
        target = target.unsqueeze(-1)
    pad_mask = target.eq(ignore_index)
    if language_tag_id is not None:
    # the prediction on the language tag is nonseless
        for id in language_tag_id:
            pad_mask |= target.eq(id)
    kl_loss = torch.nn.KLDivLoss(reduction='none')
    kd_loss1 = kl_loss(lprobs1, probs2).masked_fill_(pad_mask, 0.)
    kd_loss2 = kl_loss(lprobs2, probs1).masked_fill_(pad_mask, 0.)
    kd_loss = 0.5 * (kd_loss1 + kd_loss2).sum()
    return kd_loss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, language_tag_id=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if language_tag_id is not None:
        # the prediction on the language tag is nonseless
            for id in language_tag_id:
                pad_mask |= target.eq(id)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss



@register_criterion("label_smoothed_cross_entropy_zs")
class LabelSmoothedCrossEntropyCriterionZS(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        gamma1=0.,
        gamma2=0.,
        language_tag_id='',
        pre_train=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.pre_train = pre_train
        if language_tag_id == '':
            self.language_tag_id = None
        else:
            self.language_tag_id = []
            for item in language_tag_id.strip().split(','):
                self.language_tag_id.append(int(item) + 3)


    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--pre-train', type=bool, default=False)
        parser.add_argument("--gamma1", type=float,
                            default=0.0,
                            help="The SMD loss weight")
        parser.add_argument('--gamma2', type=float,
                            default=0.0,
                            help='the mixup example prediction loss.')
        parser.add_argument('--language-tag-id', type=str,
                            default='',
                            help='The id of language token in dictionary, start with 1, split by comma.')


    def swap_sample(self, sample):
        """
        target: no <eos>
        prev_output_tokens: first <eos>
        src_tokens: no <eos>
        default left padding
        """
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)  # append <eos>
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": (target != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens[:, :-1].contiguous()
            },
            'nsentences': sample['nsentences'],
            'ntokens': utils.item((src_tokens[:, 1:] != self.padding_idx).int().sum().data),
            "target": src_tokens[:, 1:].contiguous(),
            "id": sample["id"],
        }

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #net_output = model(**sample["net_input"])
        encoder_out = model.encoder.forward(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"])
        net_output = model.decoder.forward(sample["net_input"]['prev_output_tokens'], encoder_out)
        if not self.pre_train:
            reverse_sample = self.swap_sample(sample)
            reversed_encoder_out = model.encoder.forward(
                reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
            if self.gamma1 != 0.:
                op_loss = self.get_op_loss(encoder_out, reversed_encoder_out, sample, reverse_sample)
            else:
                op_loss = torch.tensor(0)
            if self.gamma2 != 0.:
                kd_loss = self.get_token_contrast_loss_mix(encoder_out, reversed_encoder_out, model, sample, reverse_sample)
            else:
                kd_loss = torch.tensor(0)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "op_loss": op_loss.data if not self.pre_train else 0,
            "kd_loss": kd_loss.data if not self.pre_train else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        # former
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        if not self.pre_train:
            op_loss = op_loss / nsentences * ntokens
            loss = loss + self.gamma1 * op_loss + self.gamma2 * kd_loss
        return loss, sample_size, logging_output


    def get_op_loss(self, encoder_out, reversed_encoder_out, sample, reverse_sample):
        encoder_out = encoder_out['encoder_out'][-1].transpose(0, 1)
        reversed_encoder_out = reversed_encoder_out['encoder_out'][-1].transpose(0, 1).detach().clone()
        mask1 = (sample["net_input"]["src_tokens"] == self.padding_idx)
        mask2 = (reverse_sample["net_input"]["src_tokens"] == self.padding_idx)
        op_loss1 = compute_op_distance_min(encoder_out, reversed_encoder_out, mask1, mask2)
        op_loss2 = compute_op_distance_min(reversed_encoder_out, encoder_out, mask2, mask1)
        return 0.5 * (op_loss1 + op_loss2)

    def get_token_contrast_loss_mix(self, out1, out2, model, sample1, sample2):
        out2['encoder_out'][-1] = out2['encoder_out'][-1].detach().clone()
        min_len = min(sample1["net_input"]['prev_output_tokens'].size(1),
                      sample2["net_input"]['prev_output_tokens'].size(1))
        prev1 = sample1["net_input"]['prev_output_tokens'][:, :min_len]
        prev2 = sample2["net_input"]['prev_output_tokens'][:, :min_len]
        m = torch.distributions.Beta(6,3)
        weight = m.sample(prev1.size()).type_as(prev1.float().half())
        mix_index = torch.bernoulli(weight).bool()
        mix_prev = prev1.masked_fill(~mix_index, 0) + prev2.masked_fill(mix_index, 0)
        net_out1 = model.decoder.forward(mix_prev, out1)
        net_out2 = model.decoder.forward(mix_prev, out2)
        new_out1 = net_out1[0] / 1
        new_out2 = net_out2[0] / 1
        sample = {}
        sample['target'] = sample1['target'][:, :min_len].contiguous()
        lprobs1, rand_target, probs1 = self.get_lprobs_and_target(model, [new_out1], sample)
        lprobs2, rand_target, probs2 = self.get_lprobs_and_target(model, [new_out2], sample)
        loss = compute_kd_loss(probs1, probs2, lprobs1, lprobs2, rand_target, self.padding_idx)
        return loss

    def get_lprobs_and_target(self, model, net_output, sample):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), probs.view(-1, probs.size(-1))

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target, probs = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss


    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        op_loss_sum = sum(log.get("op_loss", 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get("kd_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kd_loss", kd_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "op_loss", op_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

