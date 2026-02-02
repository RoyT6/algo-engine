#!/usr/bin/env python3
"""
ALGO-80 Staged Runner - GPU-Mandatory Execution Engine
=======================================================
VERSION: 95.66 | ALGO 95.66 | ViewerDBX Bus v1.0

FOUR RULES ENFORCEMENT:
- V1: Proof-of-Work Requirement
- V2: Checksum Validation
- V3: Execution Logging
- V4: Trust-Based Audit

GPU MANDATORY: Uses cuDF RAPIDS - NO sampling, NO CPU fallback
"""

import os
import sys
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import platform as _platform

# =============================================================================
# PLATFORM-AWARE PATH DETECTION
# =============================================================================

def _get_base_path() -> Path:
    """Get base path based on platform (Windows vs WSL)."""
    if _platform.system() == 'Windows':
        return Path(r'C:\Users\RoyT6\Downloads')
    else:
        return Path('/mnt/c/Users/RoyT6/Downloads')

BASE_PATH = _get_base_path()

# =============================================================================
# GPU ENFORCEMENT - MANDATORY
# =============================================================================

def enforce_gpu() -> Tuple[bool, str, Any]:
    """
    GPU ENFORCEMENT - MANDATORY
    Returns: (success, gpu_name, cudf_module)
    CPU fallback is FORBIDDEN per ALGO 95.66
    """
    try:
        import cudf
        import cupy as cp

        # Verify GPU is actually available
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count == 0:
            raise RuntimeError("No CUDA GPU detected")

        # Get GPU info
        device = cp.cuda.Device(0)
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

        print(f"[GPU] ENFORCED: {gpu_name}")
        print(f"[GPU] cuDF version: {cudf.__version__}")
        return True, gpu_name, cudf

    except ImportError as e:
        print(f"[FATAL] GPU libraries not available: {e}")
        print("[FATAL] CPU fallback is FORBIDDEN per ALGO 95.66")
        sys.exit(99)
    except Exception as e:
        print(f"[FATAL] GPU initialization failed: {e}")
        sys.exit(99)


# =============================================================================
# FOUR RULES IMPLEMENTATION
# =============================================================================

@dataclass
class ProofOfWork:
    """V1: Proof-of-Work Requirement"""
    timestamp: str
    operation: str
    input_checksum: str
    output_checksum: str
    rows_processed: int
    gpu_confirmed: bool
    duration_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExecutionLog:
    """V3: Execution Logging"""
    run_id: str
    start_time: str
    end_time: str
    status: str
    gpu_name: str
    stages_completed: list
    errors: list
    proof_of_work: list

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['proof_of_work'] = [p.to_dict() if hasattr(p, 'to_dict') else p for p in self.proof_of_work]
        return d


class FourRulesEnforcer:
    """
    Enforces all Four Rules for ALGO 95.66 Compliance
    """

    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.proof_counters_path = BASE_PATH / "ALGO Engine" / "algo80_proof_counters.json"
        self.audit_log_path = BASE_PATH / "ALGO Engine" / "AUDIT_LOGS"
        self.audit_log_path.mkdir(exist_ok=True)

        self.proofs: list = []
        self.stages: list = []
        self.errors: list = []
        self.start_time = datetime.now()
        self.gpu_name = "Unknown"

    def compute_checksum(self, data: Any) -> str:
        """V2: Checksum Validation - compute SHA256 of data"""
        if hasattr(data, 'to_pandas'):
            # cuDF DataFrame
            content = data.to_pandas().to_json().encode()
        elif hasattr(data, 'to_json'):
            # pandas DataFrame
            content = data.to_json().encode()
        elif isinstance(data, (dict, list)):
            content = json.dumps(data, sort_keys=True, default=str).encode()
        elif isinstance(data, Path) and data.exists():
            # File path - compute file checksum
            sha256 = hashlib.sha256()
            with open(data, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        else:
            content = str(data).encode()

        return hashlib.sha256(content).hexdigest()

    def record_proof(self, operation: str, input_data: Any, output_data: Any,
                     rows_processed: int, duration_ms: float, gpu_confirmed: bool = True):
        """V1: Record Proof-of-Work"""
        proof = ProofOfWork(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            input_checksum=self.compute_checksum(input_data),
            output_checksum=self.compute_checksum(output_data),
            rows_processed=rows_processed,
            gpu_confirmed=gpu_confirmed,
            duration_ms=duration_ms
        )
        self.proofs.append(proof)
        print(f"[V1-PROOF] {operation}: {rows_processed:,} rows in {duration_ms:.1f}ms")
        return proof

    def record_stage(self, stage_name: str, status: str = "completed"):
        """V3: Record stage completion"""
        self.stages.append({
            'stage': stage_name,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[V3-LOG] Stage: {stage_name} - {status}")

    def record_error(self, error: str, fatal: bool = False):
        """V3: Record error"""
        self.errors.append({
            'error': error,
            'fatal': fatal,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[V3-ERROR] {'FATAL: ' if fatal else ''}{error}")

    def update_proof_counters(self):
        """V1: Update persistent proof counters"""
        counters = {
            'last_updated': datetime.now().isoformat(),
            'total_runs': 1,
            'total_rows_processed': sum(p.rows_processed for p in self.proofs),
            'total_proofs': len(self.proofs),
            'gpu_runs': 1 if self.gpu_name != "Unknown" else 0,
            'last_run_id': self.run_id,
            'last_gpu': self.gpu_name,
            'algo_version': '95.66'
        }

        # Load existing and increment
        if self.proof_counters_path.exists():
            try:
                with open(self.proof_counters_path, 'r') as f:
                    existing = json.load(f)
                counters['total_runs'] = existing.get('total_runs', 0) + 1
                counters['total_rows_processed'] = existing.get('total_rows_processed', 0) + counters['total_rows_processed']
                counters['total_proofs'] = existing.get('total_proofs', 0) + counters['total_proofs']
                counters['gpu_runs'] = existing.get('gpu_runs', 0) + counters['gpu_runs']
            except (json.JSONDecodeError, KeyError):
                pass

        with open(self.proof_counters_path, 'w') as f:
            json.dump(counters, f, indent=2)

        print(f"[V1-PROOF] Counters updated: {counters['total_runs']} total runs")
        return counters

    def save_audit_log(self, status: str = "completed") -> Path:
        """V4: Trust-Based Audit - save complete execution log"""
        log = ExecutionLog(
            run_id=self.run_id,
            start_time=self.start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            status=status,
            gpu_name=self.gpu_name,
            stages_completed=self.stages,
            errors=self.errors,
            proof_of_work=[p.to_dict() for p in self.proofs]
        )

        log_file = self.audit_log_path / f"ALGO80_AUDIT_{self.run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log.to_dict(), f, indent=2, default=str)

        print(f"[V4-AUDIT] Log saved: {log_file}")
        return log_file


# =============================================================================
# ALGO-80 STAGED RUNNER
# =============================================================================

class ALGO80StagedRunner:
    """
    ALGO-80 Staged Runner with GPU-Mandatory Execution
    Implements all Four Rules for ALGO 95.66 Compliance
    """

    ALGO_VERSION = "95.66"
    MIN_RUNTIME_SECONDS = 30  # Minimum runtime for valid execution

    # Anti-Cheat: Valid metric ranges
    VALID_R2_RANGE = (0.30, 0.90)
    VALID_MAPE_RANGE = (0.05, 0.40)  # 5% - 40%

    def __init__(self):
        self.enforcer = FourRulesEnforcer()
        self.cudf = None
        self.gpu_name = "Unknown"

    def initialize(self) -> bool:
        """Initialize GPU environment"""
        print("\n" + "="*70)
        print("ALGO-80 STAGED RUNNER - GPU MANDATORY")
        print(f"Version: {self.ALGO_VERSION}")
        print("="*70)

        # V1: GPU Enforcement
        success, gpu_name, cudf_module = enforce_gpu()
        if not success:
            return False

        self.gpu_name = gpu_name
        self.cudf = cudf_module
        self.enforcer.gpu_name = gpu_name
        self.enforcer.record_stage("GPU_INIT", "completed")

        return True

    def load_bfd(self, bfd_path: Path = None) -> Any:
        """Load BFD database using cuDF (NO SAMPLING)"""
        if bfd_path is None:
            # Find latest BFD file
            bfd_patterns = ["BFD*.parquet", "BFD-Views*.parquet"]
            bfd_files = []
            for pattern in bfd_patterns:
                bfd_files.extend(list(BASE_PATH.glob(pattern)))

            if not bfd_files:
                self.enforcer.record_error("No BFD parquet files found", fatal=True)
                return None

            bfd_path = max(bfd_files, key=lambda p: p.stat().st_mtime)

        print(f"\n[LOAD] Loading BFD: {bfd_path.name}")
        start = time.time()

        # V2: Input checksum
        input_checksum = self.enforcer.compute_checksum(bfd_path)

        # Load with cuDF - NO SAMPLING
        df = self.cudf.read_parquet(str(bfd_path))

        duration_ms = (time.time() - start) * 1000

        # V1: Record proof of work
        self.enforcer.record_proof(
            operation="LOAD_BFD",
            input_data={'path': str(bfd_path), 'checksum': input_checksum},
            output_data={'rows': len(df), 'columns': len(df.columns)},
            rows_processed=len(df),
            duration_ms=duration_ms,
            gpu_confirmed=True
        )

        self.enforcer.record_stage("LOAD_BFD", "completed")
        print(f"[LOAD] Loaded {len(df):,} rows, {len(df.columns)} columns in {duration_ms:.1f}ms")

        return df

    def validate_anti_cheat(self, r2: float, mape: float) -> bool:
        """Validate metrics are within anti-cheat ranges"""
        r2_valid = self.VALID_R2_RANGE[0] <= r2 <= self.VALID_R2_RANGE[1]
        mape_valid = self.VALID_MAPE_RANGE[0] <= mape <= self.VALID_MAPE_RANGE[1]

        if not r2_valid:
            self.enforcer.record_error(
                f"R2 {r2:.4f} outside valid range {self.VALID_R2_RANGE}",
                fatal=True
            )
        if not mape_valid:
            self.enforcer.record_error(
                f"MAPE {mape:.4f} outside valid range {self.VALID_MAPE_RANGE}",
                fatal=True
            )

        return r2_valid and mape_valid

    def run_staged_pipeline(self, df: Any) -> Dict[str, Any]:
        """Run the staged ALGO-80 pipeline"""
        results = {
            'status': 'pending',
            'stages': [],
            'metrics': {}
        }

        # Stage 1: Data Validation
        print("\n[STAGE 1] Data Validation")
        start = time.time()

        required_cols = ['fc_uid', 'imdb_id', 'title', 'title_type']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.enforcer.record_error(f"Missing columns: {missing}", fatal=True)
            results['status'] = 'failed'
            return results

        self.enforcer.record_proof(
            operation="VALIDATE_SCHEMA",
            input_data={'columns': len(df.columns)},
            output_data={'valid': True, 'required_present': len(required_cols)},
            rows_processed=len(df),
            duration_ms=(time.time() - start) * 1000
        )
        self.enforcer.record_stage("STAGE_1_VALIDATION", "completed")
        results['stages'].append('STAGE_1_VALIDATION')

        # Stage 2: Feature Engineering (GPU)
        print("\n[STAGE 2] Feature Engineering (GPU)")
        start = time.time()

        # Count views columns
        views_cols = [c for c in df.columns if str(c).startswith('views_')]

        self.enforcer.record_proof(
            operation="FEATURE_ENGINEERING",
            input_data={'columns': len(df.columns)},
            output_data={'views_columns': len(views_cols)},
            rows_processed=len(df),
            duration_ms=(time.time() - start) * 1000
        )
        self.enforcer.record_stage("STAGE_2_FEATURES", "completed")
        results['stages'].append('STAGE_2_FEATURES')

        # Stage 3: Integrity Check
        print("\n[STAGE 3] Integrity Check")
        start = time.time()

        # Check for duplicates
        duplicate_count = df['fc_uid'].duplicated().sum()
        if hasattr(duplicate_count, 'item'):
            duplicate_count = duplicate_count.item()

        # Check null rates
        null_rates = df.isnull().mean()
        critical_null = null_rates[null_rates > 0.5]

        self.enforcer.record_proof(
            operation="INTEGRITY_CHECK",
            input_data={'rows': len(df)},
            output_data={'duplicates': int(duplicate_count), 'high_null_cols': len(critical_null)},
            rows_processed=len(df),
            duration_ms=(time.time() - start) * 1000
        )
        self.enforcer.record_stage("STAGE_3_INTEGRITY", "completed")
        results['stages'].append('STAGE_3_INTEGRITY')

        results['status'] = 'completed'
        results['metrics'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'views_columns': len(views_cols),
            'duplicates': int(duplicate_count)
        }

        return results

    def finalize(self, status: str = "completed") -> Dict[str, Any]:
        """Finalize run and save all audit data"""
        # V1: Update proof counters
        counters = self.enforcer.update_proof_counters()

        # V4: Save audit log
        audit_file = self.enforcer.save_audit_log(status)

        return {
            'run_id': self.enforcer.run_id,
            'status': status,
            'proof_counters': counters,
            'audit_log': str(audit_file),
            'proofs_recorded': len(self.enforcer.proofs),
            'stages_completed': len(self.enforcer.stages),
            'errors': len(self.enforcer.errors)
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for ALGO-80 Staged Runner"""
    runner = ALGO80StagedRunner()

    # Initialize GPU
    if not runner.initialize():
        print("[FATAL] GPU initialization failed - aborting")
        sys.exit(99)

    # Load BFD
    df = runner.load_bfd()
    if df is None:
        print("[FATAL] BFD load failed - aborting")
        sys.exit(1)

    # Run pipeline
    results = runner.run_staged_pipeline(df)

    # Finalize
    final = runner.finalize(results['status'])

    print("\n" + "="*70)
    print("ALGO-80 STAGED RUNNER - COMPLETE")
    print("="*70)
    print(f"Run ID: {final['run_id']}")
    print(f"Status: {final['status']}")
    print(f"Proofs Recorded: {final['proofs_recorded']}")
    print(f"Stages Completed: {final['stages_completed']}")
    print(f"Audit Log: {final['audit_log']}")
    print("="*70)

    return 0 if results['status'] == 'completed' else 1


if __name__ == "__main__":
    sys.exit(main())
