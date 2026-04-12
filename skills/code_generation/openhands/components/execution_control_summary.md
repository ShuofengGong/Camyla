# Execution Control Summary

## 🔒 Execution Control

**Python Environment:**
- Use `{python_path}` for all Python commands
- Use `{pytest_path}` for pytest

**Allowed Verifications:**
- ✅ `{python_path} -c "import experiment"` (syntax check)
- ✅ `{pytest_path} test_*.py -v` (pytest unit tests)
- ✅ `timeout --foreground -s 9 600 {python_path} test.py` (run unit test with 10-min timeout)
- ✅ `{python_path} -m py_compile experiment.py` (compile check)

**⚠️ FORBIDDEN - DO NOT RUN:**
- ❌ `python experiment.py` or any direct execution of experiment.py
- ❌ `camylanet.training_network()` (multi-epoch training)
- ❌ `camylanet.evaluate()` (requires full training)
- ❌ Any long-running operations (> 10 minutes)
- ❌ `kill`, `pkill`, `killall`, `fuser -k` or any process-termination commands (shared GPU machine)

**✅ ALLOWED for Testing:**
- ✅ `timeout --foreground -s 9 600 {python_path} test.py` (runs 1-epoch test with 10-min timeout)
- ✅ `camylanet.training_network_1epoch()` (quick validation < 10 minutes)

**Your task is to WRITE the code, not EXECUTE it. The experiment will be run separately.**

---

## Variables

| Variable | Description |
|----------|-------------|
| `python_path` | Python interpreter path |
| `pytest_path` | Pytest path |
