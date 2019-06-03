AdaNSP: Uncertainty-driven Adaptive Neural Semantic Parsing
=============

The code for our ACL19 short paper: **AdaNSP: Uncertainty-driven Adaptive Neural Semantic Parsing**.

**requirements**

- Python 3.7
- PyTorch 1.0 or greater
- AllenNLP 0.8.0 or greater

Source codes are all resided in `src` directory.
Datasets are collected from `https://github.com/donglixp/lang2logic`.

To run the training code, go to `./src/experiments` and run the following

```
PYTHONPATH=.. python unc_s2s.py -d atis
```

After training, the model and training log will be resided in `./snapshopts/atis/unc_s2s/[run-time]/`.
In order to test the model, add an `--test` option and the model file path to the command like,

```
PYTHONPATH=.. python unc_s2s.py -d atis --test ../../snapshots/atis/unc_s2s/[run-time]/model_state_epoch_[epoch_num].th
```

