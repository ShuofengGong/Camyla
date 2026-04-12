# Execution Control Rules

## 🔒 CRITICAL EXECUTION CONTROL RULES

### Python Environment Configuration

**MUST use this Python interpreter for ALL Python commands:**
```bash
{python_path}
```

**For pytest:**
```bash
{pytest_path}
```

---

## ✅ ALLOWED Operations

You ARE allowed to:

| Operation | Command | Purpose |
|-----------|---------|---------|
| **Syntax Check** | `{python_path} -c "import experiment"` | Verify imports work |
| **Unit Tests** | `{pytest_path} test_*.py -v` | Execute unit tests |
| **1-Epoch Test** | `timeout --foreground -s 9 600 {python_path} test.py` | Quick validation (10-min timeout) |
| **Type Check** | `{python_path} -m py_compile experiment.py` | Compile check |
| **Lint Check** | Code linting tools (if available) | Code quality |

### Example Allowed Commands

```bash
# Check syntax and imports
{python_path} -c "import experiment"

# Run pytest
{pytest_path} test_experiment.py -v

# Run 1-epoch unit test with 10-minute timeout
timeout --foreground -s 9 600 {python_path} test.py

# Compile check
{python_path} -m py_compile experiment.py
```

---

## ❌ FORBIDDEN Operations

You are **STRICTLY FORBIDDEN** from:

| Operation | Why Forbidden |
|-----------|---------------|
| Running `experiment.py` directly | Runs full multi-epoch training |
| `camylanet.training_network()` | Multi-epoch training (takes hours) |
| `camylanet.evaluate()` | Requires full training first |
| Any process > 10 minutes | Resource intensive |
| GPU-heavy operations | Full deep learning training |

### ❌ NEVER Run These Commands

```bash
# NEVER DO THIS
python experiment.py
{python_path} experiment.py
```

### ❌ NEVER Run Process-Management Commands

**This machine is shared with other experiments on other GPUs.** Running process-management
commands will destroy hours of work on other GPUs. This is **STRICTLY FORBIDDEN**:

```bash
# NEVER DO ANY OF THESE — they affect other experiments on shared GPUs
kill <pid>
kill -9 <pid>
pkill python
killall python
fuser -k /dev/nvidia*
fuser --kill /dev/nvidia*
nvidia-smi --gpu-reset
```

If you encounter **CUDA out of memory**, you MUST fix it by reducing model complexity
(fewer channels, gradient checkpointing, smaller layers). Do NOT attempt to free GPU
memory by terminating processes. The `timeout` command in the test workflow handles
cleanup automatically.

---

## ✅ ALLOWED for Testing

| Operation | Command | Duration | Required? |
|-----------|---------|----------|-----------|
| **1-epoch integration test** | `timeout --foreground -s 9 600 {python_path} test.py` | < 10 min | **MANDATORY** |
| Syntax verification | `{python_path} -c "import experiment"` | Instant | Optional |

**MANDATORY**: You MUST run `test.py` with `camylanet.training_network_1epoch()` before finishing. This is the only way to verify your code works with the full training pipeline (data loading, loss computation, backward pass, AMP compatibility).

---

## ⚠️ VIOLATION CONSEQUENCES

If you run forbidden operations:
- The experiment will be **corrupted**
- The task will **fail**
- Resources will be **wasted**

**Your task is to WRITE the code, not EXECUTE it. The experiment will be run separately.**

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `python_path` | Python interpreter | `/opt/conda/envs/py310/bin/python` |
| `pytest_path` | Pytest path | `/opt/conda/envs/py310/bin/pytest` |
