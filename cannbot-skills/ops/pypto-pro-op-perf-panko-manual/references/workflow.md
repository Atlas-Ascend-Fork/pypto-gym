# Base campaign workflow

Use the NPU environment’s Python, PyPTO-Pro, Optuna and CANN msprof. Keep device, imports, test inputs, shapes and tolerances fixed.
`S` is the installed scripts directory, `D` the campaign directory and `OP` the staged kernel.

## Initialize once

Use one Python script invocation that fails on precision mismatch (`TEST_COMMAND`) and a JSON list of immutable test, reference and input paths (`TEST_FILES`).
Set `CAPACITY` to the selected target's usable Vec, Mat, Left, Right and Acc budgets in bytes. If its installed tile declaration checker establishes a separate address limit for a space, pass it in bytes via `--address-limits` as a JSON object. Omitted spaces use the usable budget as the limit. Use a fresh receipt path and specify `--op-name` for ambiguous single-file profiles.
The memory objective sums source-derived high-water marks. Treat an incomplete reading as unknown.

```sh
python3 "$S/measure.py" --op-file "$OP" --op-dir "$D" --device "$DEVICE" \
  --test-command "$TEST_COMMAND" --test-files "$TEST_FILES" \
  --memory-capacity "$CAPACITY" --out "$D/baseline.json"
python3 "$S/panko_harness.py" init --op-dir "$D" --op kernel --op-file "$OP" \
  --device "$DEVICE" --config '{"eval_budget":40}' --measurement-file "$D/baseline.json"
python3 "$S/panko_harness.py" select --op-dir "$D"
```

## Verify and deliver

After STOP, run `python3 "$S/panko_harness.py" verify --auto --op-dir "$D"` for the whole Pareto front.
For manual ABBA, use `verify --op-dir "$D" --acquire` and submit its fresh receipt with `verify --op-dir "$D" --measurement-file <returned-path>` at each stage.
Run `report --op-dir "$D"`, check the restored source hash and retain the frontier sources.
Give the verifier the delivered file and hash, frozen test/reference files, command, environment, tolerances and receipts. Report success only after independent PASS.
