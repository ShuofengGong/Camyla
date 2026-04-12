# Debug Mode

## Task Overview

Your previous code had a bug that needs to be fixed. Analyze the error and make targeted fixes.

---

## 🐛 Previous Execution Error

The previous code failed with the following error:

```
{error_message}
```

---

## 📂 Existing Code Reference

**IMPORTANT**: The buggy code is in `experiment.py`.

**File Info**: `experiment.py` ({code_length} characters, {code_lines} lines)

---

## Debugging Workflow

1. **Analyze** the error message carefully
2. **Read** the existing code in `experiment.py`
3. **Identify** the root cause of the bug
4. **Make minimal, targeted fixes** - don't rewrite everything
5. **Update** `test.py` if the test itself has issues
6. **Re-run** the test to verify the fix

---

## Debugging Guidelines

### ✅ DO

- Focus on fixing the specific error
- Make minimal changes to resolve the issue
- Preserve working code - don't change what isn't broken
- Test your fix before completing

### ❌ DO NOT

- Create alternative files (e.g., `experiment_new.py`) - only `experiment.py` will be executed
{stage_specific_constraints}

---

## Common Bug Categories

| Bug Type | What to Check |
|----------|---------------|
| **ImportError** | Missing imports, typos in module names |
| **TypeError** | Wrong argument types, missing arguments |
| **ValueError** | Invalid values (e.g., wrong dimensions) |
| **AttributeError** | Using wrong attribute/method names |
| **RuntimeError** | Framework-specific issues |
| **Shape Mismatch** | Tensor dimension misalignment |

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `error_message` | Full error traceback | `"TypeError: ..."` |
| `code_length` | Characters in existing code | `12345` |
| `code_lines` | Lines in existing code | `456` |
| `stage_specific_constraints` | Stage-specific don'ts | See below |

### Stage-Specific Constraints

**Stage 1 (Baseline)**:
```
- Add optimizations or advanced features
- Implement novel techniques
- Change from baseline approach
```

**Stage 2+ (Innovation)**:
```
- Remove or skip the core innovation
- Change the fundamental innovation concept
```
