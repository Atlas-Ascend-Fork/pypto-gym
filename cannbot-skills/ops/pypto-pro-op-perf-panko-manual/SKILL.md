---
name: pypto-pro-op-perf-panko-manual
description: Base PANKO latency/on-chip-memory optimization of single-file PyPTO-Pro kernels on NPU, using catalog-guided structural search and TPE for inferred tile domains.
---

# PANKO for PyPTO-Pro

Freeze host wrappers and precision tests. Follow the [workflow](references/workflow.md) for INIT and final verification.
On resume, reconcile state, receipts and the pending directive. Keep one writer and the original budget.

For structural CODE, use the selected action’s `delta` and [profile evidence](references/world_model.md).
Give the coder the action ID, target function, edit, expected effect and correctness constraints.
Measure staged TPE candidates unchanged.

With the installed scripts (`S`), campaign (`D`), staged kernel (`OP`) and frozen test command, run STEP once per stable dispatch ID:

```sh
python3 "$S/panko_harness.py" step --op-dir "$D" --op-file "$OP" --parent "$ACTION_ID" \
  --dispatch-id "$DISPATCH_ID" --test-command "$TEST_COMMAND"
```

On STOP, finish ABBA and return `verification_request` for independent verification.
For `next=evaluate`, measure the staged tile with the same parent and a new dispatch ID. For `next=code`, request code for that parent.
For `next=close`, run `close --action "$ACTION_ID"` and check STOP before selecting again. Otherwise follow the returned `selection`.
On acquisition failure, preserve state and receipts and return ASK_USER if blocked.
