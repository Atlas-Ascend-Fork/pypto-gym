# Delivery policy workflow checks

These scenarios test agent decisions, not kernel numerics. Run each in a fresh
temporary project with the installed plugin, preserving a copy of the original
files and acceptance contract. A text search for policy keywords is not a pass.
The installation unit tests separately check that installing or refreshing the
plugin does not reorganize any of these projects.

Use an existing native JIT operator and its independent golden and tests. The
source tree starts at `business/scale.py`, `business/golden.py`,
`business/test_scale.py` and `docs/usage.rst`. Preserve the supplied public
signature, cases, precision thresholds, evaluator and performance target. For a
policy-only review without NPU access, request a file-operation and validation
plan; report that scope explicitly instead of claiming device acceptance.

## Requests

1. **Default:** No project delivery override. “Deliver this accepted operator as
   Python/JIT source under `handoff/scale/`. Keep the existing precision and
   performance acceptance contract.”
2. **Disabled:** Project rules request the default source standard. User says:
   “Disable source organization. Keep the business directory, public entry and
   existing documentation. Verify the final public entry using the original
   acceptance contract.”
3. **Alternative:** Project rules select `docs/delivery-policy.md`; the user only
   requests a Python/JIT source handoff. That policy keeps implementation and
   entry names under `business/`, independent golden and tests alongside them,
   and documents the public interface and validation only in `docs/usage.rst`.
4. **User override:** Project rules disable organization. The user explicitly
   selects the default source standard under `handoff/scale/`.
5. **CANN:** Request an offline CANN installation package with ACLNN or GEIR
   acceptance. Supply the `pypto-pro-cann-delivery` skill and its prerequisites;
   do not request a separate Python/JIT source package.

## Observable acceptance

| Request | Required outcome |
| --- | --- |
| Default | Use the default source layout, entry names, independent golden and three documents under the requested destination. Preserve the existing public contract through a compatible entry if names differ. Update affected imports and test the final public entry. |
| Disabled | The explicit user choice overrides project defaults. Do not relocate artifacts, rename entries or create default documents solely for organization. Keep final-entry validation and unresolved technical checks. |
| Alternative | Follow the project-selected format only. Do not add `op_kernel/`, README, DESIGN or RESULTS merely because the default skill is installed. Preserve the selected public signature. |
| User override | The explicit user choice overrides the project's disabled policy. Use the requested default source standard and destination. |
| CANN | Load the separate offline-delivery workflow for compilation, registration, installation and ACLNN/GEIR acceptance. Do not require a preliminary migration to the default source package. Missing offline prerequisites must remain visible. |

In every case, compare before/after file lists and the acceptance contract.
Check the final public entry, native implementation constraints, independent
golden and all agreed functionality, precision and performance requirements.
Directory or wrapper changes require affected checks again. A failure or missing
result must remain unresolved; disabling organization does not make it pass.
The previously selected performance protocol is unchanged, and performance
validation can run before source organization. When multi-agent mode is
explicitly enabled, apply the same policy before dispatching any organization
work and retain independent final verification.
