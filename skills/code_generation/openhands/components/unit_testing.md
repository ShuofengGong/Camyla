# Unit Testing Requirements

## 🚨 MANDATORY: 1-Epoch Integration Test

Before finishing, you **MUST** run a 1-epoch integration test using `camylanet.training_network_1epoch()`.
This is the **only mandatory test**. You CANNOT skip it or replace it with other tests (e.g., dummy forward pass, syntax check).

**If you finish WITHOUT passing the 1-epoch test, your implementation will be REJECTED.**

**Why this is mandatory**: A standalone forward pass with `torch.randn()` does NOT catch:
- Model output format issues (tuple vs tensor) that crash the nnUNet loss function
- Mixed precision (AMP) incompatibility (e.g., `torch.linalg.qr` on float16)
- Spatial dimension mismatches caused by pooling/upsampling at real patch sizes
- Training instability (NaN loss, gradient explosion)

Only `training_network_1epoch()` exercises the **full pipeline**: data loading → forward pass → loss computation → backward pass → optimizer step.

### test.py MUST have `if __name__ == "__main__":` guard

**This is NOT a framework bug.** Fix your test.py by wrapping all code inside `if __name__ == "__main__":`.
**Do NOT skip the 1-epoch test or replace it with a forward-pass-only test because of this error.**

### Required test.py Structure

```python
import sys
import camylanet

# Import your custom trainer from experiment.py
from experiment import dataset_id, configuration, exp_name
# Replace YourCustomTrainer with the actual trainer class name
from experiment import YourCustomTrainer

def test_1epoch_training():
    """MANDATORY: Run 1 epoch through the full camylanet pipeline"""
    plans_identifier = camylanet.plan_and_preprocess(
        dataset_id=dataset_id,
        configurations=[configuration]
    )
    
    result_folder, log = camylanet.training_network_1epoch(
        dataset_id=dataset_id,
        configuration=configuration,
        trainer_class=YourCustomTrainer,
        plans_identifier=plans_identifier,
        exp_name=f'test_{exp_name}'
    )
    
    assert log['train_losses'], "No training loss recorded"
    print(f'✅ 1-epoch training passed (loss: {log["train_losses"][-1]:.4f})')

if __name__ == '__main__':
    try:
        test_1epoch_training()
        print('✅ Mandatory test passed — implementation is ready')
        sys.exit(0)
    except Exception as e:
        print(f'❌ MANDATORY test failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

### Run Command

```bash
timeout --foreground -s 9 600 {python_path} test.py
```

## Testing Workflow

1. Create/modify `experiment.py`
2. Create `test.py` with the **mandatory 1-epoch test** above (MUST have `if __name__ == "__main__":` guard!)
3. Run: `timeout --foreground -s 9 600 {python_path} test.py`
4. If it fails → fix the code → rerun
5. **You may ONLY finish when this test passes**
6. **A standalone forward-pass test (test_model.py) does NOT substitute for this test**

## Optional Additional Tests

You may add other tests (syntax check, forward pass shape check, etc.) but they do NOT substitute for the mandatory 1-epoch test.

---
