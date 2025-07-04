import logging

from typing import List, Union
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or gtnctc
        reduce: reduce the CTC loss into a scalar
        ignore_nan_grad: Same as zero_infinity (keeping for backward compatiblity)
        zero_infinity:  Whether to zero infinite losses and the associated gradients.
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(
                reduction="none", zero_infinity=zero_infinity
            )
        elif self.ctc_type == "builtin2":
            self.ignore_nan_grad = True
            logging.warning("builtin2")
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")

        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply

        elif self.ctc_type == "brctc":
            try:
                import k2
            except ImportError:
                raise ImportError("You should install K2 to use Bayes Risk CTC")

            from espnet2.asr.bayes_risk_ctc import BayesRiskCTC

            self.ctc_loss = BayesRiskCTC(
                brctc_risk_strategy, brctc_group_strategy, brctc_risk_factor
            )

        else:
            raise ValueError(f'ctc_type must be "builtin" or "gtnctc": {self.ctc_type}')

        self.reduce = reduce
        self.mask_fn = lambda x: x

    def set_mask_fn(self, mask_fn):
        self.mask_fn = mask_fn

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen, lang_sym=None) -> torch.Tensor:
        if self.ctc_type == "builtin" or self.ctc_type == "brctc":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.ctc_type == "builtin":
                size = th_pred.size(1)
            else:
                size = loss.size(0)  # some invalid examples will be excluded

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        # builtin2 ignores nan losses using the logic below, while
        # builtin relies on the zero_infinity flag in pytorch CTC
        elif self.ctc_type == "builtin2":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens, return_output=False, lang_sym=None):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        ys_hat = self.mask_fn(ys_hat)
        self.ctc_out = ys_hat

        if self.ctc_type == "brctc":
            loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens).to(
                device=hs_pad.device, dtype=hs_pad.dtype
            )
            return loss

        elif self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )
        if return_output:
            return loss, self.ctc_out
        return loss

    def softmax(self, hs_pad, lang_sym=None, task_label=None):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        ctc_out = self.ctc_lo(hs_pad)
        return F.softmax(self.mask_fn(ctc_out), dim=2)

    def log_softmax(self, hs_pad, lang_sym=None, task_label=None):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        ctc_out = self.ctc_lo(hs_pad)
        return F.log_softmax(self.mask_fn(ctc_out), dim=2)

    def argmax(self, hs_pad, lang_sym=None, task_label=None):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        ctc_out = self.ctc_lo(hs_pad)
        return torch.argmax(self.mask_fn(ctc_out), dim=2)

class MultilingualCTC(torch.nn.Module):
    def __init__(
        self,
        language_ids: List[Union[str, int]],
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.reduce = reduce
        self.mask_fn = lambda x: x

        # Create language-specific output layers
        self.ctc_lo = torch.nn.ModuleDict({
            str(lang_id): torch.nn.Linear(encoder_output_size, odim)
            for lang_id in language_ids
        })

        # Shared CTC loss function
        if ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(reduction="none", zero_infinity=zero_infinity)
        elif ctc_type == "builtin2":
            self.ignore_nan_grad = True
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")
        elif ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction
            self.ctc_loss = GTNCTCLossFunction.apply
        elif ctc_type == "brctc":
            import k2  # required
            from espnet2.asr.bayes_risk_ctc import BayesRiskCTC
            self.ctc_loss = BayesRiskCTC(brctc_risk_strategy, brctc_group_strategy, brctc_risk_factor)
        else:
            raise ValueError(f"Unsupported ctc_type: {ctc_type}")

        self.ctc_type = ctc_type

    def _get_proj(self, lang_sym: Union[str, int]):
        lang_key = lang_sym
        assert lang_key in self.ctc_lo, f"CTC projection for lang_sym '{lang_sym}' not found, lang_sym must be one of {self.ctc_lo.keys()}"
        return self.ctc_lo[lang_key]

    def forward(self, hs_pad, hlens, ys_pad, ys_lens, return_output=False, lang_sym=None):
        proj = self._get_proj(lang_sym)
        ys_hat = proj(F.dropout(hs_pad, p=self.dropout_rate))
        ys_hat = self.mask_fn(ys_hat)
        self.ctc_out = ys_hat

        if self.ctc_type == "brctc":
            loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens)
            return (loss, self.ctc_out) if return_output else loss

        if self.ctc_type == "gtnctc":
            ys_true = [y[y != -1] for y in ys_pad]
        else:
            ys_hat = ys_hat.transpose(0, 1)  # (B, T, D) → (T, B, D)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens)
        return (loss, self.ctc_out) if return_output else loss

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        th_pred = th_pred.log_softmax(2)
        loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

        if self.ctc_type == "builtin":
            size = th_pred.size(1)
        elif self.ctc_type == "brctc":
            size = loss.size(0)
        else:
            size = th_pred.size(1)

        if self.reduce:
            return loss.sum() / size
        return loss / size

    def log_softmax(self, hs_pad, lang_sym=None, task_label=None):
        proj = self._get_proj(lang_sym)
        ctc_out = proj(hs_pad)
        return F.log_softmax(self.mask_fn(ctc_out), dim=2)

    def softmax(self, hs_pad, lang_sym=None):
        proj = self._get_proj(lang_sym)
        ctc_out = pro(hs_pad)
        return F.softmax(self.mask_fn(ctc_out), dim=2)

    def argmax(self, hs_pad, lang_sym=None, task_label=None):
        proj = self._get_proj(lang_sym)
        ctc_out = proj(hs_pad)
        return torch.argmax(self.mask_fn(ctc_out), dim=2)

