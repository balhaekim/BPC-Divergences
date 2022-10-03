# On Divergence Measures for Bayesian Pseudocoresets
This repository is the official implementation of [On Divergence Measures for Bayesian Pseudocoresets](https://openreview.net/login?redirect=%2Fforum%3Fid%3Dbg7d_2jWv6) (NeurIPS 2022)
### Generating Expert Trajectories
Before training any Bayesian pseudocoreset, you'll need to generate some expert trajectories using `buffer.py`
```python
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100
```
### Training Bayesian pseudocoresets
The following command will then use the buffers we just generated to train Bayesian pseudocoresets of each divergence measure:
```python
python train.py --dataset=CIFAR10 --model=ConvNet --divergence={fkl, rkl, wasserstein} --ipc={1, 10, 20} --eval_method={hmc, sghmc}
```

## Acknowledgments
Our code is adapted from [https://github.com/GeorgeCazenavette/mtt-distillation](https://github.com/GeorgeCazenavette/mtt-distillation)

## Citation
If you find this useful in your research, please consider citing our paper:
```
@inproceedings{kim2022pseudocoresets,
  title     = {On Divergence Measures for Bayesian Pseudocoresets},
  author    = {Balhae Kim and Jungwon Choi and Seanie Lee and Yoonho Lee and Jung-Woo Ha and Juho Lee},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```
