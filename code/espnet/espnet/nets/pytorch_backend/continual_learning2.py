# Copyright 2025 Steven Vander Eeckt

""" Continual Learning extension for End-to-End ASR (ESPnet2) """
import torch
import logging
from abc import ABC, abstractmethod
import os
from espnet2.fileio.read_text import load_num_sequence_text
import torch.nn.functional as func
import copy
import typing
import sentencepiece as spm
import random

from espnet2.layers.create_adapter_utils import (
    check_target_module_exists,
    get_submodules,
    replace_module,
    get_target_key,
)
from espnet2.torch_utils.model_summary import model_summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################
########################### HELP FUNCTIONS ##########################################################################
#####################################################################################################################


def distillation_loss(
        teacher_output: torch.tensor,
        student_output: torch.tensor,
        dim: int = 2,
        neg: bool = True,
        teacher_softmax: bool = True,
        divide_by_batch_size: bool = True,
        per_sample: bool = False
):
    """
        Distillation loss between teacher and student

        :param torch.Tensor teacher_output: 3D (2D) logits of teacher
        :param torch.Tensor student_output: 3D (2D) logits of student
        :param int dim: dimension along which to apply softmax
        :param bool neg: if True, -loss is returned, else loss
        :param bool teacher_softmax: if False, teacher_output is already softmaxed
        :param bool divide_by_batch_size: if True, loss is divded by batch size
        :param bool per_sample: if True, loss is returned per mini-batch
    """
    sm_t, lsm_s = torch.nn.Softmax(dim=dim), torch.nn.LogSoftmax(dim=dim)
    prob_t = sm_t(teacher_output) if teacher_softmax else teacher_output
    log_s = lsm_s(student_output)
    if not per_sample:
        loss = (prob_t * log_s).sum(dim=dim).sum()
    else:
        loss = (prob_t * log_s).sum(dim=dim).sum(dim=-1)
    # check if there is a batch size
    if divide_by_batch_size and len(teacher_output.size()) > 2:
        # if there is, divide loss by it
        loss = loss / teacher_output.size(0)
    return -loss if neg else loss


def _recursive_to(
        xs,
        device
):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple) or isinstance(xs, list):
        return tuple(_recursive_to(x, device) for x in xs)
    if isinstance(xs, dict):
        return {n: _recursive_to(p, device) for n, p in xs.items()}
    return xs

def clone_module(module):
    # Create a new instance of the same class as the original module
    cloned_module = type(module)()

    # Copy the parameters from the original module to the new module
    for original_param, cloned_param in zip(module.parameters(), cloned_module.parameters()):
        cloned_param.data = original_param.data.clone()

    return cloned_module

def copy_model(
        model: torch.nn.Module,
        device: str,
        train: bool = False,
        requires_grad: bool = False,
):
    """
        Copy the model and make it untrainable

        :param torch.nn.Module model: the model
        :param str device:  cpu or gpu device to map model to
        :param bool train: if False, model is set to eval mode
        :param bool requires_grad:
    """
    # for batch norm and dropout layers
    old_model = copy.deepcopy(model).train(train)
    for name, param in old_model.named_parameters():
        param.requires_grad = requires_grad
    return old_model.to(device)


def item(t):
    try:
        return t.item()
    except:
        return t


#####################################################################################################################
########################### ABSTRACT CLASSES ########################################################################
#####################################################################################################################


"""
ABSTRACT CLASSES FOR CL METHODS: HIERARCHY
    -- CL_Method
        -- LossBasedCL: "Add a regularization term to the ASR loss"
        
        -- Rehearsal:       "Use a memory of samples from previous tasks"
            -- KnowledgeDistillationBased:   "Uses KD on a memory to distill knowledge from old to new model"
        
        -- DataFocused:     "LWF-like regularization-based methods, using the new task's mini-batch"
        
        -- SVDBasedCL:   "Decomposes linear layers using SVD and trains only a 
                                gating vector corresponding to singular values"            
"""


class CLMethod(ABC):
    """
        CL Method Abstract Class
    """
    pass


class LossBasedCL(CLMethod):
    """
        Loss-Based CL Abstract Class

        Includes all methods, regularization and rehearsal, which enable CL by means of a regularizing loss

        :method compute_loss: each class inheriting LossBasedCL must have a compute_loss(model) method
    """
    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the loss given the model
            :param torch.nn.Module model:
        """
        pass


class Rehearsal(CLMethod):
    """
        Rehearsal Abstract Class
        Generic class to be inherited by Rehearsal-based CL methods

        :param str device:
        :param str task_file : task file to map utterances to tasks (task IDs)
        :param str task_order : sets the task IDs in the right order (e.g. '1 2 0 3' means that task with task ID = 1 is
                                the first task, task with task ID = 2 the second, and so on)
    """
    def __init__(
            self,
            device: str,
            task_file: str = "",
            task_order: str = "",
            group: bool = False,
            memory_text: str = "",
            num_tokens: int = 5000,
    ):
        self.device = device
        # check if model has task specific part
        if group:
            assert num_tokens > 0
            assert memory_text is not None
            self.get_sp = lambda lang: spm.SentencePieceProcessor(model_file=f'data/{lang}_token_list/bpe_unigram{num_tokens}/bpe.model')
            self.other_sp = {}
            self.utt2text_ids = {}
            self.utt2text = self._prepare_utt2text(memory_text)
            self.get_batch = lambda return_names=False: self._get_batch_and_lang(return_names)
        elif task_file and task_order:
            # load utterance to task mapping
            logging.info("opening task file.. %s" % task_file)
            utt2task = load_num_sequence_text(task_file, loader_type="csv_int")
            assert task_file != ""
            # load task ID ordering
            task_order = {int(task): i for i, task in enumerate(task_order.split(" "), 0)}
            # dicts to map utterance and batches to tasks
            self.utt2task = {utt: task_order[task[0]] for utt, task in utt2task.items()}
            self.batch2task = {}
            # set the default get_batch() function.
            self.get_batch = lambda return_names=False: self._get_batch_and_task(return_names)
        else:
            # set the default get_batch() function.
            self.get_batch = lambda return_names=False: self._get_batch(return_names)

    def set_loader(
            self,
            loader
    ):
        """
            Set the loader and iterator of the rehearsal-based method

            :param torch.Dataloader loader: the dataloader for the memory of the rehearsal-based method
        """
        self.loader = loader.build_iter(epoch=0, shuffle=True)
        self.iter = iter(self.loader)

    @staticmethod
    def get_unique_key(
            names: typing.List[str],
    ):
        """
            Generates unique key for batch with utterances in names

            :param List[str] names: list of utterance IDs of batch
        """
        return "_".join(sorted(names))

    def _get_batch(
            self,
            return_names: bool = False
    ):
        """
            Sample a batch from the iterator (memory)
        """
        try:
            #names, xs = self.iter.next()
            names, xs = next(self.iter)
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            #names, xs = self.iter.next()
            names, xs = next(self.iter)
        if return_names:
            return _recursive_to(xs, self.device), names
        return  _recursive_to(xs, self.device)

    def _get_batch_and_lang(
            self,
            return_names: bool = False,
    ):
        """
            Sample a batch from the iterator (memory)
        """
        try:
            names, xs = next(self.iter)
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            names, xs = next(self.iter)
        names, xs = self._convert_batch(names, xs)
        if return_names:
            return _recursive_to(xs, self.device), names
        return  _recursive_to(xs, self.device)

    def _convert_batch(
            self,
            names: typing.List[str],
            batch: typing.Dict,
    ):
        key = self.get_unique_key(names)
        # if utterance has already been converted once
        try:
            lang, _ = self._get_lang_and_sp_model(names)
            text, text_lengths = self.utt2text_ids[key]
        except KeyError as e:
            # get lang and corresponding sp model
            old_text, old_text_lengths = batch['text'], batch['text_lengths']
            # get text and text_lengths in old alphabet
            text, text_lengths = old_text.clone(), old_text_lengths.clone()
            # get lang and sp_model of new alphabet
            lang, sp_model = self._get_lang_and_sp_model(names)
            # convert text
            new_text, new_text_lengths = [], []
            for i in range(old_text.size(0)):
                # convert text to sequence for correct alphabet
                t = sp_model.encode(self.utt2text[names[i]], out_type=int)
                # substract 1 (because we added 1s before)
                t = [j - 1 for j in t]
                # add substract 1 and append -1s
                t_ = torch.tensor(t, device=self.device)
                # replace text[i] by t_
                new_text.append(t_)
                # update text_lengths
                new_text_lengths.append(len(t))
            # Determine max length
            max_len = max(new_text_lengths)
            pad_idx = -1
            padded_texts = torch.full((len(new_text), max_len), fill_value=pad_idx, device=self.device)
            for i, t in enumerate(new_text):
                padded_texts[i, :t.size(0)] = t
            # Set final text and length tensors
            text = padded_texts
            text_lengths = torch.tensor(new_text_lengths, dtype=torch.long, device=self.device)
            # store in utt2text
            self.utt2text_ids[key] = (text, text_lengths)
        batch['text'], batch['text_lengths'], batch['lang_sym'] = text, text_lengths, lang
        return names, batch

    def _get_lang_and_sp_model(
            self,
            names: typing.List[str],
    ):
        def get_lang(utt_key):
            speaker_id = utt_key.split("-")[1]
            lang = speaker_id.split("_")[2]
            return lang
        langs = [get_lang(name) for name in names]
        assert len(list(set(langs))) == 1, f"Found more than one language: {list(set(langs))}"
        lang = langs[0]
        if not lang in self.other_sp.keys():
            self.other_sp[lang] = self.get_sp(lang)
        return lang, self.other_sp[lang]

    @staticmethod
    def _prepare_utt2text(
            memory_text: str,
    ):
        # read memory text and store in utt2gt (ground truth)
        utt2gt = {}
        with open(memory_text, "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = line.strip("\n")
                utt, text = line.split(" ")[0], " ".join(line.split(" ")[1:])
                utt2gt[utt] = text
                line = f.readline()
        return utt2gt

    def get_task_from_batch(
            self,
            names: typing.List[str]
    ):
        """
            Given (the utterance IDs of a batch), return the task number

            :param List[str] names: list containing the utterance IDs of the batch
        """
        # generate a unique hashable key
        key = self.get_unique_key(names)
        # select task from batch2task if it exists:
        try:
            task = self.batch2task[key]
            return task
        except Exception as e:  # the first time we encounter this batch
            tasks = set([self.utt2task[name] for name in names])
            assert len(tasks) == 1, "batch %s had multiple tasks: %s" % (str(names), str(tasks))
            self.batch2task[key] = list(tasks)[0]
            return list(tasks)[0]

    def _get_batch_and_task(
            self,
            return_names: bool = False
    ):
        """
            Sample a batch from the iterator (memory)
            and add the task to the batch.

            :param bool return_names: return utterance IDs
        """
        try:
            names, xs = self.iter.next()
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            names, xs = self.iter.next()
        # get the task of the batch
        task = self.get_task_from_batch(names)
        # add it to the batch
        xs['task_label'] = task
        if return_names:
            return _recursive_to(xs, self.device), names
        return _recursive_to(xs, self.device)


class KnowledgeDistillationBased(LossBasedCL, Rehearsal):
    """
        Knowledge Distillation-based Rehearsal Abstract Class

        :param str device: the device for the model
        :param float alpha: the weight of the regularization
        :param float ctc_weight: the weight of CTC in the KL div loss
        :param float temperature: temperature of the KL div loss
        :param bool eval_mode: if True, old model (which generates logits) is in eval mode
        :param str task_file: see Rehearsal
        :param str task_order: see Rehearsal
    """
    def __init__(
            self,
            device: str,
            alpha: float = 1.0,
            ctc_weight: float = 0.3,
            task_file: str = "",
            task_order: str = "",
            memory_text: str = "",
            num_tokens: int = 5000,
            group: bool = False,
    ):
        super(KnowledgeDistillationBased, self).__init__(device=device, task_file=task_file, task_order=task_order,
                                                         memory_text=memory_text, num_tokens=num_tokens,
                                                         group=group)
        self.ctc_weight = ctc_weight
        self.alpha = alpha
        self.T = 1.0
        self.KLdiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def distillation_loss(
            self,
            teacher_output: torch.tensor,
            student_output: torch.tensor,
    ):
        """
            Computes the distillation loss
            :param torch.Tensor teacher_output:
            :param torch.Tensor student_output:
        """
        input, target = func.log_softmax(student_output, dim=2), func.log_softmax(teacher_output, dim=2)
        return self.KLdiv(input, target)

    def set_model(
            self,
            model: torch.nn.Module
    ):
        """
            Sets the old model

            :param torch.nn.Module model: the old model
        """
        self.old_model = copy_model(model, self.device, train=False)

    def set_loader(
            self,
            loader
    ):
        """
            Set the loader and iterator of the rehearsal-based method

            :param torch.Dataloader loader: the dataloader for the memory of the rehearsal-based method
        """
        self.loader = loader.build_iter(epoch=0, shuffle=True)
        self.iter = iter(self.loader)
        # if self.eval_mode == True, output of model is deterministic
        # therefore, we can already compute the output of the memory by the old model
        self.old_outputs = self.collect_outputs()

    def collect_outputs(
            self
    ):
        """
            In case old_model is in eval_mode, outputs are deterministic and they can be computed before training
            That way, they don't have to be computed at each step.
        """
        # to store the outputs
        old_outputs = {}
        # sample over all batches in memory
        while True:
            # sample a batch - we also need the utterance IDs
            batch, names = self.get_batch(return_names=True)
            #logging.info("Collecting outputs for batch from memory: %d (size = %s)" % (len(old_outputs) + 1,
            #                                                                           str(batch['speech'].size())))
            # if already present in old_outputs, we have processed all batches
            if self.get_unique_key(names) in old_outputs.keys():
                break
            # model must return the output
            batch['return_args'] = ["decoder_out", "ctc_out"]
            # compute output from old model
            _, old_dec_output, old_ctc_output = self.old_model.forward_er(**batch)
            old_outputs[self.get_unique_key(names)] = {'ctc': old_ctc_output.detach().cpu(),
                                                       'dec': old_dec_output.detach().cpu()}
        return old_outputs

    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        pass


class DataFocused(LossBasedCL):
    """
        Data-Focused CL Abstract class

        Generic class to be inherited by LWF and similar methods

        :param str device:
        :param float alpha:
        :param float ctc_weight:
        :param int temperature:
        :param bool eval_mode:
    """
    def __init__(
          self,
          device: str,
          alpha: float,
          ctc_weight: float = 0.0,
          temperature: float = 1,
    ):
        logging.info("DataFocused with: ctc_weight = %.2f, alpha = %.2f, T = %.2f" % (ctc_weight, alpha, temperature))
        self.ctc_weight = ctc_weight
        self.alpha = alpha
        self.T = 1.0 * temperature
        self.device = device
        self.old_model = None

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        """
          Sets the old model

          :param torch.nn.Module model: the old model
        """
        self.old_model = copy_model(model, self.device, train=False)

    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: typing.Dict = None,
            decoder_output: torch.tensor = None,
            ctc_output: torch.tensor = None
    ):
        """
          :param torch.nn.Module model:
          :param torch.tensor inputs: input for the encoder (xs_pad, src_mask, -1, True)
          :param torch.tensor decoder_output: current output of the decoder
          :param torch.tensor ctc_output: current output of CTC
        """
        pass


class SVDBasedCL(CLMethod):
    """
        SVDBasedCL - decomposes linear layers of model as W + U(alpha*S)V^T where USV^T = delta W, the difference
                between the current parameters and the previous parameters.

        :param str init_param: path to state dict with parameters of the old models
        :param str outdir: current output directory to store sigmoid(alphas)
        :param str old_tasks: number of old tasks, to multiply loss with (t-1)
    """
    def __init__(
            self,
            init_param: str,
            outdir: str,
    ):
        if isinstance(init_param, list) or isinstance(init_param, tuple):
            init_param = init_param[0]
        # load state dict of old model
        try:
            logging.info(f"Loading old model: {init_param}")
            self.old_model = torch.load(init_param)
        except FileNotFoundError as fnfe:
            logging.warning(fnfe)
            self.old_model = {}
        self.outdir = outdir

    def set_avg_bias(
            self,
            current_bias: torch.nn.Parameter,
            old_bias: torch.Tensor,
    ):
        """
        Set the current bias parameter based on the average of new and old bias.

        Args:
            current_bias (torch.nn.Parameter): The current bias parameter to be updated.
            old_bias (torch.Tensor): The old bias tensor to use for updating.
        """
        # Ensure shapes match
        assert current_bias.size() == old_bias.size(), "Size mismatch between current_bias and old_bias"
        avg_bias = (current_bias.data + old_bias.to(current_bias.device)) / 2
        current_bias.data.copy_(avg_bias)

    def set_model(
            self,
            model: torch.nn.Module,
            optimizers: typing.List[torch.optim.Optimizer],
    ):
        """
        Applies the SVD-Based decomposition to linear layers of the given model

          :param torch.nn.Module model: the model
          :param torch.optim.Optimizers optimizers: to add the alphas to the optimizer for optimization
        """
        # do nothing if eval mode
        if not model.train:
            return
        class SVDLinear(torch.nn.Linear):
            """
                SVDLinear module: writes a linear layers as W + U(alpha*S)V' where USV' = delta W, the update applied
                    to the linear layer compared to the old model.

            """
            def __init__(
                    self,
                    in_features: int,
                    out_features: int,
                    **kwargs
            ):
                torch.nn.Linear.__init__(self, in_features, out_features, **kwargs)
                self.U, self.S, self.Vh, self.alpha, self.M = None, None, None, None, None
                self.sigmoid = None

            def reset_parameters(self):
                torch.nn.Linear.reset_parameters(self)

            def prepare_svd(self, old_weight):
                """
                    Prepares the SVD-decomposition, based on the current weight matrix and on the old weight matrix

                    :param torch.Tensor old_weight: the weight matrix from the old model
                """
                weight_diff = self.weight - old_weight.to(self.weight.device)
                U, S, Vh = torch.linalg.svd(weight_diff, full_matrices=False)
                # set old_weight as self.weight
                self.weight.data.copy_(old_weight)
                # freeze weight
                self.weight.requires_grad = False
                if self.bias is not None:
                    self.bias.requires_grad = False
                # define and freeze U, S, Vh
                self.U = torch.nn.Parameter(U, requires_grad=False)
                self.S = torch.nn.Parameter(S, requires_grad=False)
                self.Vh = torch.nn.Parameter(Vh, requires_grad=False)
                # introduce learnable parameter alpha: sigmoid(alpha) approx 0.0 at the start
                self.alpha = torch.nn.Parameter(torch.ones_like(S) * -10.00, requires_grad=True)
                self.sigmoid = lambda x: func.sigmoid(x)

            def extra_repr(self):
                s = f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}; \n'
                """Extra representation of the module to include adapter information."""
                k = min(self.in_features, self.out_features)
                s += f'   U = ({self.in_features}, {k}), S = ({k}, {k}), alpha = ({k}), Vh = ({k}, {self.out_features})'
                return s

            def train(self, mode: bool = True):
                torch.nn.Linear.train(self, mode)

            def eval(self):
                torch.nn.Linear.eval(self)

            def state_dict(self, destination=None, prefix='', keep_vars=False):
                """Override state_dict to only store the reconstructed weight and bias."""
                state = super().state_dict(destination, prefix, keep_vars)
                # Reconstruct the full weight matrix
                full_weight = self.weight + self.U @ torch.diag(self.sigmoid(self.alpha) * self.S) @ self.Vh
                # Replace entries in state_dict
                state[prefix + 'weight'] = full_weight
                # Remove U, S, Vh from state_dict
                state.pop(prefix + 'U', None)
                state.pop(prefix + 'S', None)
                state.pop(prefix + 'Vh', None)
                state.pop(prefix + 'alpha', None)
                return state

            def forward(self, x: torch.Tensor):
                assert self.U is not None or self.S is not None or self.Vh is not None
                result = torch.nn.functional.linear(x, self.weight +
                                                    self.U @ torch.diag(self.sigmoid(self.alpha) * self.S) @ self.Vh,
                                                    bias=self.bias)
                return result

        # keep track of replaced layers
        replaced_layers = set()
        # replace every linear layer with an SVDLinear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                # create new module
                new_module = SVDLinear(module.in_features, module.out_features)
                parent_module, target_name, target_module = get_submodules(model, name)
                replace_module(parent_module, target_name, module, new_module, copy_weight=True)
                old_weight = self.old_model[name + ".weight"]
                new_module.prepare_svd(old_weight)
                self.set_avg_bias(module.bias, self.old_model[name + '.bias'])
                replaced_layers.add(name)
        # Update parameters for layers that were **not** replaced, i.e. non-torch.nn.Linear layers
        for name, param in model.named_parameters():
            # Extract the module name by removing the parameter suffix
            module_name = ".".join(name.split(".")[:-1])
            # If the layer was not replaced, apply averaging
            if module_name not in replaced_layers:
                old_param = self.old_model[name]
                param.data.copy_((old_param.to(param.device) + param) / 2)
        # update param groups of optimizers[0]
        alphas = [p for p in model.parameters() if p.requires_grad and p is not None]
        optimizers[0].param_groups.clear()
        optimizers[0].add_param_group({'params': alphas})
        logging.info(model_summary(model))

    def update_model(self, model: torch.nn.Module):
        """
            Store sigmoid(alpha)'s values in current output directory
        """
        alphas = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'alpha'):
                alpha = module.sigmoid(module.alpha)
                alphas[name] = alpha
        torch.save(alphas, f"{self.outdir}/alphas.pth")



#####################################################################################################################
########################### REHEARSAL-BASED CL METHODS ##############################################################
#####################################################################################################################

"""
    Rehearsal-Based CL methods: overview
        -- ER: "trains jointly on mini-batch from current task and one from memory"
        -- KD: "teacher-student method with the old model the teacher and current model the student, using data from memory" 
        -- SVR: "efficient rehearsal training only gating vectors corresponding to singular values of adaptation"

"""
class ER(LossBasedCL, Rehearsal):
    """
        Experience Replay for ASR model.

        :param str device: the device for the model and data
        :param float alpha: the weight of the regularization
    """
    def __init__(
            self,
            device: str,
            alpha: float = 1,
            task_file: str = "",
            task_order: str = "",
            memory_text: str = "",
            num_tokens: int = 5000,
            group: bool = False,
    ):
        super(ER, self).__init__(device=device, task_file=task_file, task_order=task_order,
                                 memory_text=memory_text, num_tokens=num_tokens, group=group)
        self.alpha = alpha
        logging.info("CL Method: ER with alpha=%.2f" % (alpha))

    def compute_loss(
            self,
            model: torch.nn.Module
    ):
        """
            Computes the ER loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch
        batch = self.get_batch()
        # compute loss on new model
        loss = model.forward_er(**batch)
        # multiply by regularization weight
        return self.alpha * loss


class KD(KnowledgeDistillationBased):
    """
        Knowledge Distillation with memory for Continual Learning

        :param str device: the device for the model
        :param float alpha: the weight of the regularization
        :param float ctc_weight: the weight of CTC in the KL div loss
        :param str task_file: related to memory
        :param str task_oder: related to memory
        :param str memory_text: related to memory
        :param str group: related to memory
    """

    def __init__(
            self,
            device: str,
            alpha: float = 1.0,
            ctc_weight: float = 0.3,
            eval_mode: bool = True,
            task_file: str = "",
            task_order: str = "",
            memory_text: str = "",
            num_tokens: int = 5000,
            group: bool = False,
    ):
        super(KD, self).__init__(device=device, alpha=alpha, ctc_weight=ctc_weight,
                                 task_file=task_file, task_order=task_order, num_tokens=num_tokens,
                                 memory_text=memory_text, group=group)
        logging.info("KD with alpha = %.2f, temperature = %.1f, eval_mode = %s" % (alpha, 1.0, str(eval_mode)))


    def compute_loss(
            self,
            model: torch.nn.Module
    ):
        """
            Computes the KD loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch from memory
        x, names = self.get_batch(return_names=True)
        # model should return output, not loss
        x['return_args'] = ['decoder_out', 'ctc_out']
        # obtain output from current and old model
        _, dec_output, ctc_output = model.forward_er(**x)
        # if outputs have already been computed, retrieve them:
        if hasattr(self, 'old_outputs'):
            # return the unique key
            key = self.get_unique_key(names)
            # return the old dec and ctc outputs, computed previously
            old_dec_output, old_ctc_output = (self.old_outputs[key]['dec'].to(self.device),
                                              self.old_outputs[key]['ctc'].to(self.device))
        # otherwise compute them now
        else:
            _, old_dec_output, old_ctc_output = self.old_model.forward_er(**x)
        # compute KL div loss at decoder-level
        loss = self.distillation_loss(old_dec_output / self.T, dec_output / self.T)
        # optionally, compute KL div loss at CTC-level
        if self.ctc_weight > 0:
            loss = (self.ctc_weight * self.distillation_loss(old_ctc_output / self.T, ctc_output / self.T)
                    + (1 - self.ctc_weight) * loss)
        # multiply KD loss by regularization weight
        return self.alpha * loss


class SVR(KD, SVDBasedCL):
    def __init__(
            self,
            device: str,
            outdir: str,
            old_init_param: str,
            ctc_weight: float = 0.3,
            task_file: str = "",
            task_order: str = "",
            memory_text: str = "",
            num_tokens: int = 5000,
            group: bool = False,
            old_tasks: int = 1,  # to multiply loss with (t-1)

    ):
        KD.__init__(self, device=device, alpha=1.0, ctc_weight=ctc_weight, task_file=task_file, task_order=task_order,
                                 memory_text=memory_text, num_tokens=num_tokens, group=group)
        SVDBasedCL.__init__(self, outdir=outdir, init_param=old_init_param)
        # SKD must only have KD loss
        self.new_model = None
        self.old_tasks = old_tasks

    def set_model(
            self,
            model: torch.nn.Module,
            optimizers: typing.List[torch.optim.Optimizer],
    ):
        """
            Applies SVD-Based decomposition to current model and sets the old model for KD (to transfer knowledge from)

            :param torch.nn.Module model: the old model
        """
        # apply SVD-Based decomposition
        SVDBasedCL.set_model(self, model, optimizers=optimizers)
        self._old_model = copy_model(model, self.device, train=False)
        replaced_layers = set()
        # Set the old model for KD to transfer knowledge from
        # linear layers in old model must have: alpha = 0, so that W + U(alpha*S)V' = W (old param)
        no_svdlinear_layers = True
        for name, module in self._old_model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and hasattr(module, 'alpha'):
                # alpha_init: such that sigmoid(gamma * M  @ alpha) = 0.0
                y = torch.full((module.alpha.size(0),), -10.0, device=module.alpha.device)  # shape: (d,)
                module.alpha.data.copy_(y)
                replaced_layers.add(name)
        # non-torch.nn.Linear parameters must get their old values for old model in KD
        for name, param in self._old_model.named_parameters():
            # Extract the module name by removing the parameter suffix
            module_name = ".".join(name.split(".")[:-1])
            # If the layer was not replaced, apply averaging
            if module_name not in replaced_layers:
                old_param = self.old_model[name]
                param.data.copy_(old_param.to(param.device))
                logging.info(f"Using old params {name}...")
            else:
                logging.info(f"NOT using old params {name}...")
        self.old_model = self._old_model

    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the KD loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch from memory
        x, names = self.get_batch(return_names=True)
        # compute er-loss and kd-loss
        x['return_args'] = ['decoder_out', 'ctc_out']
        # obtain output from current and old model
        er_loss, dec_output, ctc_output = model.forward_er(**x)
        # if outputs have already been computed, retrieve them:
        if hasattr(self, 'old_outputs'):
            # return the unique key
            key = self.get_unique_key(names)
            # return the old dec and ctc outputs, computed previously
            old_dec_output, old_ctc_output = (self.old_outputs[key]['dec'].to(self.device),
                                              self.old_outputs[key]['ctc'].to(self.device))
        # otherwise compute them now
        else:
            _, old_dec_output, old_ctc_output = self.old_model.forward_er(**x)
        # compute KL div loss at decoder-level
        kd_loss = self.distillation_loss(old_dec_output / self.T, dec_output / self.T)
        # optionally, compute KL div loss at CTC-level
        if self.ctc_weight > 0:
            kd_loss = (self.ctc_weight * self.distillation_loss(old_ctc_output / self.T, ctc_output / self.T)
                    + (1 - self.ctc_weight) * kd_loss)
        reh_loss = self.alpha * (kd_loss + er_loss) / 2
        stats = {"kd_loss": item(kd_loss), "er_loss": item(er_loss)}
        return self.old_tasks * reh_loss, stats


#####################################################################################################################
########################### REGULARIZATION-BASED CL METHODS #########################################################
#####################################################################################################################

class SVR_NoRegularization(SVDBasedCL):
    """
        Experience Replay extension for ASR model.

        :param str device: the device for the model and data
        :param float alpha: the weight of the regularization
    """
    def __init__(
            self,
            outdir: str,
            old_init_param: str,

    ):
        SVDBasedCL.__init__(self, outdir=outdir, init_param=old_init_param)


#####################################################################################################################
########################### DATA-FOCUSED CL METHODS #################################################################
#####################################################################################################################

"""
    Data-focused CL methods:
        -- LWF: "uses new task's data to transfer knowledge from old model (teacher) to current model (student)"
"""


class LWF(DataFocused):
    """
        Learning without Forgetting for Encoder-Decoder ASR model
        :param float alpha: weight of the regularization
        :param float ctc_weight: weight for CTC
    """
    def __init__(
            self,
            device: str,
            alpha: float,
            ctc_weight: float = 0.0,
    ):
        super(LWF, self).__init__(device, alpha, ctc_weight)

    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: typing.Dict = None,
            decoder_output: torch.tensor = None,
            ctc_output: torch.tensor = None
    ):
        """
          Compute the loss term of the LWF regularizer
          :param torch.nn.Module model:
          :param dict inputs: contains keys [speech, speech_lengths, text, text_lengths, task]
          :param torch.tensor decoder_output: current output of the decoder
          :param torch.tensor ctc_output: current output of CTC
        """
        inputs["return_args"] = ["ctc_out", "decoder_out"]
        # send inputs through old_model
        _, old_ctc_output, old_decoder_output = self.old_model.forward_er(**inputs)
        # compute Decoder loss term
        loss = distillation_loss(old_decoder_output / self.T, decoder_output / self.T)
        # compute CTC loss term
        if self.ctc_weight > 0.0:
            loss = (self.ctc_weight * distillation_loss(old_ctc_output / self.T, ctc_output / self.T)
                  + (1 - self.ctc_weight) * loss)
        return self.alpha * loss

class UOE(CLMethod):
    def __init__(
            self,
            device: str,
            **kwargs,
    ):
        self.device = device

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
        for name, param in model.named_parameters():
            if not 'encoders' in name:
                param.requires_grad_(False)
        logging.info(model_summary(model))


class CLRLTuning(CLMethod):
    def __init__(
            self,
            device: str,
            K=1
    ):
        super().__init__()
        self.device = device
        self.K = K
        self.num_encs = 0

    def get_encoders_to_unfreeze(self):
        selected_encoders = random.sample(range(self.num_encs), self.K)
        logging.info(f"Freezing encoders {sorted(selected_encoders)}")
        names = [f'encoder.encoders.{i}.' for i in selected_encoders]
        def freeze(layer_name):
            for name in names:
                if name in layer_name:
                    return False
            return True
        return freeze

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        # set the number of encoders
        self.num_encs = len(model.encoder.encoders)
        # first set to trainable all parameters
        for name, param in model.named_parameters():
            param.requires_grad_(True)
        # freeze decoder
        for name, param in model.named_parameters():
            if not 'encoders' in name:
                param.requires_grad_(False)
        # freeze all except K encoders
        freeze_func = self.get_encoders_to_unfreeze()
        for name, param in model.named_parameters():
            if freeze_func(name):
                param.requires_grad_(False)
        logging.info(model_summary(model))

    def update_model(
            self,
            model: torch.nn.Module
    ):
        """
            Called after each epoch to update the weights of the model
        """
        # freeze all except K encoders
        freeze_func = self.get_encoders_to_unfreeze()
        for name, param in model.named_parameters():
            if freeze_func(name):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        logging.info(model_summary(model))

