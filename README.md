# Efficient Rehearsal for Continual Learning in ASR via Singular Value Tuning
Supplementary material to the paper "Efficient Rehearsal for Continual Learning in ASR via Singular Value Tuning", submitted to IEEE TASLP.

## code

ESPnet2 [Wanatabe et al., 2018] is used for all experiments. 

The `code` directory contains all scripts that were added to ESPnet2 or adjusted to implement SVR and the CL baselines. Adding these files to an ESPnet2 installation should allow users to run the methods. 

Note in particular the script `continual_learning2.py`, which implements the CL methods, which are then added to the model in `espnet2/asr/espnet_model.py`.

## data

For all training, validation, test and memory sets, this directory contains the list of utterances (and speakers).

The sets are structured by experiment and task. 

For Experiment 4 (`data/exp4`), tasks `nl` and `vl`, in addition to the lists of utterances and speakers, contains the wav.scp and segments files, since we did not use the "original" utterances but merged utteranes into longer sequences of 30s. 

## models

This directory contains the configuration files for both training (of SVR and CL baselines) and decoding, for the model trained from scratch from Exp. 1-3 and OWSM v3.2 [Peng et al., 2024] from Exp. 4.  

## References 

[Kudo and Richardson, 2018] Taku Kudo and John Richardson. SentencePiece: A simple and language independentsubword tokenizer and detokenizer for neural text processing. InProceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics.

[Peng et al., 2024] Y. Peng et al., “Owsm v3.1: Better and faster open whisper-style speech models based on e-branchformer,” in Interspeech, pp. 352–356, 2024.

[Strik et al., 2000] Helmer Strik, Catia Cucchiarini, and Judith M. Kessens, “Comparing the recognition performance of csrs: in search of an adequate metric and statistical significance test,” in INTERSPEECH, 2000.

[Vander Eeckt and Van hamme, 2022] S. Vander Eeckt and H. Van hamme, “Continual learning for monolingual end-to-end automatic speech recognition,” Proceedings EUSIPCO 2022, 2022.

[Watanabe et al., 2018] S. Watanabe et al., “ESPnet: End-to-end speech processing toolkit,” in Proceedings of Interspeech, 2018, pp. 2207–2211. 



