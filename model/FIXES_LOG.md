# FIXES_LOG.md

Running log of every change made to get the following command working cleanly
on Python 3.10 / PyTorch 2.14+cpu / NumPy 2.x / RDKit 2026.x (2022 codebase):

```
venv\Scripts\python.exe train.py `
  --data_path ./toy_dataset/novel_pair_0_train.csv `
  --separate_val_path ./toy_dataset/novel_pair_0_val.csv `
  --separate_test_path ./toy_dataset/novel_pair_0_test.csv `
  --metric mse --dataset_type regression `
  --save_dir test_run --target_columns label `
  --epochs 5 --ensemble_size 1 --num_folds 1 `
  --batch_size 5 --aggregation mean --dropout 0.1 --save_preds
```

---

## Fix 1 — Missing package: tap (typed-argument-parser)

- **Error**: ModuleNotFoundError: No module named 'tap'
- **File**: chemprop/args.py (import at top of module)
- **Root cause**: typed-argument-parser was not in the venv.
- **Fix**: pip install typed-argument-parser (done by user before this log started)
- **Why**: Required package for TrainArgs/PredictArgs argument parsing classes.

---

## Fix 2 — Missing package: lifelines

- **Error**: ModuleNotFoundError: No module named 'lifelines'
- **File**: chemprop/train/run_training.py, line 24 (from lifelines.utils import concordance_index)
- **Root cause**: lifelines was not in the venv.
- **Fix**: pip install lifelines (installs lifelines 0.30.0 and dependencies)
- **Why**: Used to compute the concordance index (C-index) metric at the end of training.

---

## Fix 3 — Hardcoded .cuda() in model.py (3 sites)

- **Error**: AssertionError: Torch not compiled with CUDA enabled
- **File**: chemprop/models/model.py
- **Lines**: 41, 216, 238
- **Root cause**: Three unconditional .cuda() calls that crash when running on CPU-only PyTorch.

### Line 41 — self.scale tensor:
BEFORE: self.scale = torch.sqrt(torch.FloatTensor([args.alpha])).cuda()
AFTER:  self.register_buffer('scale', torch.sqrt(torch.FloatTensor([args.alpha])))
(register_buffer makes scale a persistent tensor that automatically moves with the model)

### Line 216 — sequence tensor in forward():
BEFORE: sequence = sequence_tensor.cuda()
AFTER:  sequence = sequence_tensor.to(next(self.parameters()).device)

### Line 238 — add_feature tensor in forward():
BEFORE: add_feature = self.do(self.relu(self.fc_mg(add_feature.cuda())))
AFTER:  add_feature = self.do(self.relu(self.fc_mg(add_feature.to(next(self.parameters()).device))))

---

## Fix 4 — Hardcoded .cuda() in CAB.py (1 site)

- **Error**: AssertionError: Torch not compiled with CUDA enabled
- **File**: chemprop/models/CAB.py
- **Line**: 32
- **Root cause**: AttentionBlock.__init__ set self.scale with .cuda().

BEFORE: self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
AFTER:  self.register_buffer('scale', torch.sqrt(torch.FloatTensor([hid_dim // n_heads])))
(Same rationale as Fix 3 line 41)

---

## Fix 5 — torch.load default weights_only=True in PyTorch 2.6+ (4 sites)

- **Error**: _pickle.UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL argparse.Namespace
- **File**: chemprop/utils.py
- **Lines**: 106, 195, 276, 309
- **Root cause**: PyTorch 2.6 changed torch.load default from weights_only=False to weights_only=True.
  The checkpoints store argparse.Namespace objects which are not in the default safe-globals allowlist.

BEFORE: torch.load(path, map_location=lambda storage, loc: storage)
AFTER:  torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)

Applied to all four call sites: load_checkpoint (line 106), load_frzn_model (line 195),
load_scalers (line 276), load_args (line 309).
The checkpoints are produced by this codebase itself (trusted source), so weights_only=False is safe.

---

## Summary Table

| # | File | Line(s) | Error type | Fix |
|---|------|---------|------------|-----|
| 1 | pip install typed-argument-parser | - | Missing package | pip install |
| 2 | pip install lifelines | - | Missing package | pip install |
| 3 | chemprop/models/model.py | 41, 216, 238 | Unconditional .cuda() | register_buffer + .to(device) |
| 4 | chemprop/models/CAB.py | 32 | Unconditional .cuda() | register_buffer |
| 5 | chemprop/utils.py | 106, 195, 276, 309 | torch.load API change (PyTorch 2.6+) | Add weights_only=False |

No model architecture, hyperparameters, or training logic was changed.
