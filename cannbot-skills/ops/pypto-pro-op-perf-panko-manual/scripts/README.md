# Pro runtime

Pro initialization copies the shared PyPTO skill’s `scripts/` into the installed Pro skill’s `scripts/_core/`. Rerun initialization after a core update.
At runtime, `panko_harness.py` registers `pro_backend.py`. Imports through `panko_core` use `bootstrap.core_path()` to find the installed copy or the sibling PyPTO scripts in a checkout.

The core handles search, Pareto acceptance, rollback, STOP and ABBA delivery. Pro supplies its action catalog, legal tile and address rewrites, source staging, NPU measurements and source-derived memory usage.

Run the host tests from the repository root:

```sh
python3 -m unittest discover -s cannbot-skills/plugins-official/pypto-pro-op-orchestrator/tests
```
