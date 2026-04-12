# Standard Code Generation Guidelines

## ⚠️ File Operation Rules
- **Workspace Only**: All file edits MUST be within the workspace directory.
- **No External Edits**: DO NOT modify files outside the workspace. Import libraries normally, never edit their source.
- **Final Deliverable**: Only `experiment.py` will be executed. Files like `experiment_new.py`, `experiment_v2.py` etc. will be **IGNORED**. Always fix the original file.

## 🚨 Main Guard Requirement
**CRITICAL**: Wrap all executable code in `main()` with `if __name__ == "__main__":` guard.

### Correct Structure
```python
import camylanet

# Configuration
dataset_id = ... 
configuration = ...
exp_name = ...

def main():
    # All executable code here
    camylanet.training_network(...)

if __name__ == "__main__":
    main()
```
**WHY**: Without this guard, `import experiment` runs all code immediately, corrupting the experiment process.

## 🧪 MANDATORY 1-Epoch Integration Test (CANNOT BE SKIPPED OR REPLACED)
**CRITICAL**: Before finishing, you MUST pass a 1-epoch integration test using `camylanet.training_network_1epoch()`.
This is the **only mandatory test**. Standalone forward-pass tests (e.g., `torch.randn()` → `model(x)`) do **NOT** count — they miss output format, AMP, and training stability issues.

**If you finish WITHOUT passing the 1-epoch test, your implementation will be REJECTED.**

### test.py MUST have `if __name__ == "__main__":` guard

⚠️ **CRITICAL**: `test.py` MUST wrap all executable code inside `if __name__ == "__main__":`. Without this guard, you will get a **multiprocessing spawn error**:

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**This is NOT a framework bug — it means your test.py is missing the `__main__` guard. Fix your test.py, do NOT skip or replace the 1-epoch test.**

### Required test.py Template

```python
import sys
import camylanet
from experiment import dataset_id, configuration, exp_name
from experiment import YourCustomTrainer  # Replace with actual trainer class name

def test_1epoch():
    plans_identifier = camylanet.plan_and_preprocess(
        dataset_id=dataset_id, configurations=[configuration]
    )
    result_folder, log = camylanet.training_network_1epoch(
        dataset_id=dataset_id,
        configuration=configuration,
        trainer_class=YourCustomTrainer,
        plans_identifier=plans_identifier,
        exp_name=f'test_{exp_name}'
    )
    assert log['train_losses'], "No training loss recorded"
    print(f'✅ 1-epoch test passed (loss: {log["train_losses"][-1]:.4f})')

if __name__ == '__main__':
    try:
        test_1epoch()
        print('✅ Mandatory test passed — implementation is ready')
        sys.exit(0)
    except Exception as e:
        print(f'❌ MANDATORY test failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

### Testing Workflow
1. Edit `experiment.py`
2. Create `test.py` following the **required template above** (with `if __name__ == "__main__":` guard!)
3. Run: `timeout --foreground -s 9 600 {{ python_path }} test.py`
4. If fail → fix the code → rerun
5. **You may ONLY finish when this 1-epoch test passes**
6. **A standalone forward-pass test (test_model.py) does NOT substitute for this test**

## 🧠 CUDA OOM Debugging

If 1-epoch test fails with **CUDA out of memory**, do NOT simplify or remove modules. OOM means your code has a memory bug.

**Step 1: Find the bottleneck** — Add this to `test.py` BEFORE the 1-epoch call:
```python
import torch
_hooks = []
for name, mod in model.named_modules():
    def _h(n):
        def hook(m, i, o):
            print(f"[MEM] {n}: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
        return hook
    _hooks.append(mod.register_forward_hook(_h(name)))
```
Run and check which module causes the spike.

**Step 2: Common culprits and fixes**:
| Symptom | Cause | Fix |
|---------|-------|-----|
| Spike at attention/transformer | Full spatial self-attention on 3D feature maps (N×N attention matrix where N=D×H×W can be >10000) | Use **windowed** attention (window_size=4→8), attention per window is only 64×64 |
| Spike at convolution | `torch.eye(...).repeat(...)` or dynamic weight creation in forward() | Use `nn.Conv3d` with fixed weights, never create weight tensors in forward() |
| Gradual growth | Storing all intermediate tensors | Use `torch.utils.checkpoint` for memory-heavy encoder blocks |

**Step 3: If the bug cannot be isolated**, you may reduce complexity as a last resort:
- Reduce channel dimensions or number of attention heads (acceptable)
- Use gradient checkpointing for memory-heavy blocks (preferred)
- Do NOT remove innovation modules entirely — reducing their internal size is OK, removing them is NOT

**Step 4: Verify** — After fixing, peak memory should be <24 GB. Re-run 1-epoch test.

**NEVER** respond to OOM by replacing innovation modules with plain Conv3d/MLP. Reducing their internal dimensions is fine; removing them removes scientific value.

## 🔒 Execution Control Summary
- **Python Path**: `{{ python_path }}`
- **Test Command**: `timeout --foreground -s 9 600 {{ python_path }} test.py`
- **Allowed**: Syntax checks, 1-epoch tests (≤10min)
- **Forbidden**: Full `experiment.py` execution, training >10min

**Your task is to WRITE the code. Full experiment runs happen separately.**
