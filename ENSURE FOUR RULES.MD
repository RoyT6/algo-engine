# ENSURE FOUR RULES - ALGO 95.2 Compliance
## Version 22.02 | Updated 2026-01-18

---

## THE FOUR RULES

### V1: ENSURE Proof-of-Work Requirement
**Vulnerability Blocked:** System accepting file existence as proof of ML execution
**Enforcement:** Must verify actual work was performed, not just output exists

### V2: ENSURE Checksum Validation
**Vulnerability Blocked:** Identical files to previous version not flagged
**Enforcement:** SHA-256 + MD5 checksums on all I/O, output MUST differ from input

### V3: ENSURE Execution Logging
**Vulnerability Blocked:** Operator claiming pipeline ran without execution trace
**Enforcement:** Cryptographic chain of timestamped log entries

### V4: ENSURE Trust-Based Audit
**Vulnerability Blocked:** Audit log accepting operator-written entries without verification
**Enforcement:** Machine-generated audit entries ONLY, operator_editable=false

---

## BUG FIX APPLICATION (2026-01-18)

### Bug #1: Star Schema Missing Temporal Dimension

| Rule | Enforcement | Implementation |
|------|-------------|----------------|
| V1 | StarSchemaValidator checks actual columns | `validate_structure()` returns `has_temporal_required` |
| V2 | Input/output checksums registered | `register_checksum()` with category="INPUT_STAR" |
| V3 | All operations logged with hashes | `_log_operation("STAR_SCHEMA_VALIDATION", ...)` |
| V4 | Audit entry machine-generated | `create_audit_entry()` with `machine_generated=True` |

### Bug #2: Season 5 Views in 2023

| Rule | Enforcement | Implementation |
|------|-------------|----------------|
| V1 | Validator checks every row | `rows_checked` count logged, must be > 0 |
| V2 | Input BFD checksum != Output BFD checksum | Fixes change data, checksums must differ |
| V3 | Violations logged with full context | `fc_uid`, `column`, `column_year`, `start_year`, `invalid_value` |
| V4 | Violation counts machine-generated | `violations_found` in audit entry, not operator editable |

### Bug #2b: Scattered Zero Values

| Rule | Enforcement | Implementation |
|------|-------------|----------------|
| V1 | Zero detection scans all views columns | `validate_zero_vs_null()` iterates all rows |
| V2 | N/A (reporting only, fix requires confirmation) | User must confirm before changes |
| V3 | Warnings logged separately from violations | `warnings` array in results |
| V4 | Suspicious zeros flagged automatically | No manual intervention in detection |

---

## ENFORCEMENT SCRIPT: temporal_views_validator.py

### Four Rules Implementation

```python
class FourRulesEnforcer:
    """
    V1: verify_proof_of_work() - Checks metrics like rows_checked > 0
    V2: register_checksum() - SHA-256 checksums in registry
    V3: _log_operation() - Cryptographic chain with prev_hash
    V4: create_audit_entry() - machine_generated=True, operator_editable=False
    """
```

### Execution Flow

```
1. Initialize FourRulesEnforcer (V3: creates session_id, genesis log)
2. Load input files (V2: compute and register checksums)
3. Run validation (V1: verify rows_checked > 0)
4. Generate fix scripts (V3: log all operations)
5. Create audit entry (V4: machine-generated only)
6. Finalize (V3: complete chain, save to AUDIT_LOGS/)
```

### Output Files

| File | Purpose | Four Rules |
|------|---------|------------|
| `AUDIT_LOGS/temporal_validation_*.json` | Complete execution log | V3 |
| `temporal_validation_results_*.json` | Summary with violations | V1, V4 |
| `fix_bfd_temporal.py` | Auto-generated fix script | V3 (logged) |
| `fix_star_schema_temporal.py` | Auto-generated fix script | V3 (logged) |

---

## VALIDATION GATES

### Pre-Execution
- [ ] Circuit breaker state = CLOSED
- [ ] Input files exist and are readable
- [ ] Checksums computable (not locked/corrupted)

### During Execution
- [ ] V1: rows_checked incrementing (work happening)
- [ ] V3: execution_log growing (operations logged)

### Post-Execution
- [ ] V2: Output checksum != Input checksum (if fixes applied)
- [ ] V4: Audit entry has machine_generated=True
- [ ] V3: Execution log chain is valid (each hash references prev)

---

## RALPH LOOP INTEGRATION

```bash
# Run validation with Ralph monitoring
cd "/mnt/c/Users/RoyT6/Downloads/ALGO Engine"
ralph --monitor --calls 50 --prompt PROMPT.md

# Ralph will:
# 1. Read @fix_plan.md and see Bug #1, #2 tasks
# 2. Execute temporal_views_validator.py
# 3. Log results per Four Rules
# 4. Generate fix scripts
# 5. Report violations in RALPH_STATUS block
```

### Ralph Exit Conditions for Bug Fixes

```
EXIT_SIGNAL: true when:
- All Bug #1, #2 violations = 0
- Fix scripts executed successfully
- Output checksums differ from input
- Audit chain verified
- MAPIE report generated with 0 temporal violations
```

---

## SCHEMA UPDATES REQUIRED

Apply `SCHEMA_V22.01_PATCH.json` to add:

1. **rule_temporal**: Views columns must be NULL if year < start_year
2. **rule_null_vs_zero**: 0.0 = measured zero, NULL = no measurement
3. **star_schema_temporal**: period_type, period_year columns mandatory

---

**Last Updated**: 2026-01-18
**ALGO Version**: 95.2
**Ralph Version**: v0.9.9
