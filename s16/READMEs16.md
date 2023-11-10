# S16: Speeding up Transformer training

### Implementations:

1. English to French dataset
2. Automatic Mixed Precision
3. Dynamic padding (using `pad_sequence` and no manual pad count calculations)
4. One Cycle Policy
5. Parameter sharing
6. Scaled Dot Product Attention (SDP Kernel context)
7. Added code for Gradient Accumulation
8. Converted encoder_input and decoder_input Tensors in `_ _ getitem_ _` from `int64` to `int32` to reduce some memory (label in int32 gave error, need to look into loss function)
9. optimizer.zero_grad(set_to_none=True)
10. `.to(device, non_blocking=True)` (where applicable)
11. Was able to use `batch_size = 64 (batch size = 72 with 256 d_model)`
12. Able to get `per epoch time` in 2nd run to 5 mins and 10+ seconds (Colab free version T4 GPU).

#### Defaults:

```python
{'batch_size': 32,
 'num_epochs': 10,
 'lr': 0.001,
 'max_lr': 0.01,
 'pct_start': 0.1,
 'initial_div_factor': 10,
 'final_div_factor': 10,
 'anneal_strategy': 'linear',
 'three_phase': True,
 'seq_len': 500,
 'd_model': 512,
 'lang_src': 'en',
 'lang_tgt': 'fr',
 'model_folder': 'weights',
 'model_basename': 'tmodel_',
 'preload': False,
 'tokenizer_file': 'tokenizer_{0}.json',
 'experiment_name': 'runs/tmodel',
 'enable_amp': True,
 'd_ff': 512,
 'N': 6,
 'h': 8,
 'param_sharing': True,
 'gradient_accumulation': False,
 'accumulation_steps': 4}
```



#### Final 1: config changes

```python
cfg['batch_size'] = 64
cfg['preload'] = False
cfg['num_epochs'] = 90
cfg['d_model'] = 512
cfg['d_ff'] = 128
cfg['pct_start'] = 0.1
cfg['max_lr'] = 15**-3
cfg['initial_div_factor'] = 20
cfg['final_div_factor'] = 30
# cfg['gradient_accumulation'] = True
# cfg['gradient_accumulation_steps'] = 40

from train import train_model

train_model(config=cfg)
```



#### Run logs:

```
Max len of source sentence: 471
Max len of target sentence: 482

⚡️⚡️Number of model parameters: 56,337,490
Processing Epoch: 00:   0%|          | 0/1788 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:136: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
Processing Epoch: 00: 100%|██████████| 1788/1788 [09:20<00:00,  3.19it/s, loss=4.01090, lr=[4.609247865726642e-05]]
Processing Epoch: 01: 100%|██████████| 1788/1788 [09:21<00:00,  3.19it/s, loss=3.10540, lr=[7.737014249971804e-05]]
Processing Epoch: 02: 100%|██████████| 1788/1788 [09:23<00:00,  3.18it/s, loss=2.45699, lr=[0.00010864780634216965]]
Processing Epoch: 03: 100%|██████████| 1788/1788 [09:22<00:00,  3.18it/s, loss=2.82286, lr=[0.00013992547018462124]]
Processing Epoch: 04: 100%|██████████| 1788/1788 [09:21<00:00,  3.19it/s, loss=2.25028, lr=[0.00017120313402707284]]
Processing Epoch: 05: 100%|██████████| 1788/1788 [09:20<00:00,  3.19it/s, loss=2.27817, lr=[0.00020248079786952446]]
Processing Epoch: 06: 100%|██████████| 1788/1788 [09:24<00:00,  3.17it/s, loss=2.71856, lr=[0.00023375846171197606]]
Processing Epoch: 07: 100%|██████████| 1788/1788 [09:23<00:00,  3.18it/s, loss=2.35653, lr=[0.0002650361255544277]]
Processing Epoch: 08: 100%|██████████| 1788/1788 [09:25<00:00,  3.16it/s, loss=2.41267, lr=[0.00029627880319571325]]
Processing Epoch: 09: 100%|██████████| 1788/1788 [09:23<00:00,  3.18it/s, loss=2.10282, lr=[0.00026500113935326163]]
Processing Epoch: 10: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=2.14193, lr=[0.00023372347551081006]]
Processing Epoch: 11: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=1.92109, lr=[0.00020244581166835844]]
Processing Epoch: 12: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=1.76329, lr=[0.00017116814782590681]]
Processing Epoch: 13: 100%|██████████| 1788/1788 [09:28<00:00,  3.15it/s, loss=2.00005, lr=[0.00013989048398345522]]
Processing Epoch: 14: 100%|██████████| 1788/1788 [09:27<00:00,  3.15it/s, loss=2.01738, lr=[0.00010861282014100362]]
Processing Epoch: 15: 100%|██████████| 1788/1788 [09:27<00:00,  3.15it/s, loss=1.85546, lr=[7.7335156298552e-05]]
Processing Epoch: 16: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=1.61865, lr=[4.6057492456100376e-05]]
Processing Epoch: 17: 100%|██████████| 1788/1788 [09:28<00:00,  3.14it/s, loss=1.67664, lr=[1.4814592330406225e-05]]
Processing Epoch: 18: 100%|██████████| 1788/1788 [09:29<00:00,  3.14it/s, loss=1.50942, lr=[1.4615691269126826e-05]]
Processing Epoch: 19: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=1.67826, lr=[1.4416790207847428e-05]]
Processing Epoch: 20: 100%|██████████| 1788/1788 [09:31<00:00,  3.13it/s, loss=1.84702, lr=[1.421788914656803e-05]]
Processing Epoch: 21: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=1.64628, lr=[1.4018988085288631e-05]]
Processing Epoch: 22: 100%|██████████| 1788/1788 [09:26<00:00,  3.16it/s, loss=1.68622, lr=[1.3820087024009233e-05]]
Processing Epoch: 23: 100%|██████████| 1788/1788 [09:22<00:00,  3.18it/s, loss=1.64516, lr=[1.3621185962729835e-05]]
Processing Epoch: 24:  93%|█████████▎| 1662/1788 [08:51<00:40,  3.12it/s, loss=1.55427, lr=[1.3436190176987307e-05]]
```



#### Final 2 with Gradient Accumulation : config changes

```python
cfg['batch_size'] = 72
cfg['preload'] = False
cfg['num_epochs'] = 30
cfg['d_model'] = 256
cfg['d_ff'] = 128
cfg['pct_start'] = 0.2
cfg['max_lr'] = 10**-3
cfg['initial_div_factor'] = 10
cfg['final_div_factor'] = 10
cfg['gradient_accumulation'] = True
cfg['gradient_accumulation_steps'] = 40

from train import train_model

train_model(config=cfg)
```



#### Training logs:

```
Max len of source sentence: 471
Max len of target sentence: 482

⚡️⚡️Number of model parameters: 25,824,850

Processing Epoch: 00: 100%|██████████| 1589/1589 [05:10<00:00,  5.12it/s, loss=3.10041, lr=[0.00025001573481590265]]
Processing Epoch: 01: 100%|██████████| 1589/1589 [05:08<00:00,  5.15it/s, loss=3.09889, lr=[0.0004000314696318053]]
Processing Epoch: 02: 100%|██████████| 1589/1589 [05:09<00:00,  5.14it/s, loss=2.68248, lr=[0.000550047204447708]]
Processing Epoch: 03: 100%|██████████| 1589/1589 [05:08<00:00,  5.16it/s, loss=2.03687, lr=[0.0007000629392636107]]
Processing Epoch: 04: 100%|██████████| 1589/1589 [05:06<00:00,  5.18it/s, loss=2.23102, lr=[0.0008500786740795132]]
Processing Epoch: 05: 100%|██████████| 1589/1589 [05:08<00:00,  5.15it/s, loss=2.10534, lr=[0.000999905591104584]]
Processing Epoch: 06: 100%|██████████| 1589/1589 [05:10<00:00,  5.11it/s, loss=1.94918, lr=[0.0008498898562886814]]
Processing Epoch: 07: 100%|██████████| 1589/1589 [05:09<00:00,  5.13it/s, loss=1.85274, lr=[0.0006998741214727787]]
Processing Epoch: 08: 100%|██████████| 1589/1589 [05:08<00:00,  5.15it/s, loss=1.81905, lr=[0.0005498583866568761]]
Processing Epoch: 09: 100%|██████████| 1589/1589 [05:09<00:00,  5.14it/s, loss=1.83035, lr=[0.00039984265184097353]]
Processing Epoch: 10: 100%|██████████| 1589/1589 [05:10<00:00,  5.12it/s, loss=1.79879, lr=[0.00024982691702507087]]
Processing Epoch: 11: 100%|██████████| 1589/1589 [05:09<00:00,  5.14it/s, loss=1.75192, lr=[9.999370695381603e-05]]
Processing Epoch: 12: 100%|██████████| 1589/1589 [05:09<00:00,  5.13it/s, loss=1.79108, lr=[9.499388176065449e-05]]
Processing Epoch: 13: 100%|██████████| 1589/1589 [05:10<00:00,  5.12it/s, loss=1.68849, lr=[8.999405656749292e-05]]
Processing Epoch: 14: 100%|██████████| 1589/1589 [05:08<00:00,  5.15it/s, loss=1.84601, lr=[8.499423137433136e-05]]
Processing Epoch: 15: 100%|██████████| 1589/1589 [05:07<00:00,  5.17it/s, loss=1.80378, lr=[7.999440618116981e-05]]
Processing Epoch: 16: 100%|██████████| 1589/1589 [05:09<00:00,  5.13it/s, loss=1.95736, lr=[7.499458098800825e-05]]
Processing Epoch: 17: 100%|██████████| 1589/1589 [05:10<00:00,  5.12it/s, loss=1.71087, lr=[6.999475579484669e-05]]
Processing Epoch: 18: 100%|██████████| 1589/1589 [05:11<00:00,  5.10it/s, loss=1.71832, lr=[6.499493060168514e-05]]
Processing Epoch: 19:  34%|███▍      | 543/1589 [01:46<03:24,  5.12it/s, loss=1.72798, lr=[6.328636856273818e-05]]
```



#### Resuming training:

```python
cfg['batch_size'] = 72
cfg['preload'] = True
cfg['num_epochs'] = 30
cfg['d_model'] = 256
cfg['d_ff'] = 128
cfg['pct_start'] = 0.2
cfg['max_lr'] = 10**-3
cfg['initial_div_factor'] = 10
cfg['final_div_factor'] = 10

cfg['gradient_accumulation'] = True
cfg['gradient_accumulation_steps'] = 40

from train import train_model

train_model(config=cfg)
```



#### logs:

```
Max len of source sentence: 471
Max len of target sentence: 482

⚡️⚡️Number of model parameters: 25,824,850

Preloading model weights/tmodel_20.pt
Preloaded
Processing Epoch: 21: 100%|██████████| 1589/1589 [05:15<00:00,  5.04it/s, loss=1.76465, lr=[0.00025001573481590265]]
Processing Epoch: 22: 100%|██████████| 1589/1589 [05:10<00:00,  5.12it/s, loss=2.01428, lr=[0.0004000314696318053]]
Processing Epoch: 23: 100%|██████████| 1589/1589 [05:09<00:00,  5.14it/s, loss=1.82859, lr=[0.000550047204447708]]
Processing Epoch: 24: 100%|██████████| 1589/1589 [05:11<00:00,  5.11it/s, loss=1.74023, lr=[0.0007000629392636107]]
Processing Epoch: 25: 100%|██████████| 1589/1589 [05:13<00:00,  5.07it/s, loss=1.73396, lr=[0.0008500786740795132]]
Processing Epoch: 26: 100%|██████████| 1589/1589 [05:11<00:00,  5.11it/s, loss=1.80579, lr=[0.000999905591104584]]
Processing Epoch: 27: 100%|██████████| 1589/1589 [05:11<00:00,  5.10it/s, loss=1.89716, lr=[0.0008498898562886814]]
Processing Epoch: 28: 100%|██████████| 1589/1589 [05:10<00:00,  5.11it/s, loss=1.77185, lr=[0.0006998741214727787]]
Processing Epoch: 29: 100%|██████████| 1589/1589 [05:12<00:00,  5.09it/s, loss=1.65273, lr=[0.0005498583866568761]]
--------------------------------------------------------------------------------
    SOURCE: The cavalcade was brilliant, and its march resounded on the pavement.
    TARGET: La cavalcade était brillante et résonnait sur le pavé.
 PREDICTED: La cavalcade était brillante et sa marche s ' agitait sur le pavé .
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
    SOURCE: There was nothing extraordinary about the country; the sky was blue, the trees swayed; a flock of sheep passed.
    TARGET: Mais non! la campagne n’avait rien d’extraordinaire: le ciel était bleu, les arbres se balançaient; un troupeau de moutons passa.
 PREDICTED: Le ciel était bleu , des arbres , des cris de mouflons , un troupeau de mouflons passa .
```

