#!/usr/bin/env python3
"""Audit the public computational export without running archived code.

Standard library only; does not retrain models or create new measurement data.
Checks CRCs, hashes, unique members/records, budgets, and published L8 medians.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


EXPECTED_PHYSICAL = {
    (0.125, 10**8): (24.523, 21.884, 17.724, 17.724),
    (0.125, 10**9): (11.205, 9.620, 8.400, 8.400),
    (0.25, 10**8): (28.070, 30.240, 29.514, 29.874),
    (0.25, 10**9): (17.538, 21.658, 27.937, 23.811),
}
METHODS = ('raw_cutoff', 'Hlag_cutoff', 'diagonal_cv', 'general_cv_including_diagonal')


def verify(root: Path) -> dict:
    exports = json.loads((root / 'data/EXPORTS.json').read_text(encoding='utf-8'))
    report = {'status': 'PASS', 'scope': 'release integrity and recorded-metric checks; no new simulation',
              'archives': [], 'l8_physical_medians_percent': []}
    for spec in exports:
        path = root / 'data' / spec['file']
        require(path.is_file(), f'Missing archive: {path.name}')
        require(path.stat().st_size == spec['bytes'], f'Size mismatch: {path.name}')
        require(file_sha256(path) == spec['sha256'], f'ZIP SHA256 mismatch: {path.name}')
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), f'Duplicate members: {path.name}')
            for name in names:
                require(not name.startswith('/') and '..' not in Path(name).parts,
                        f'Unsafe member path: {name}')
            require(archive.testzip() is None, f'CRC error: {path.name}')
            manifest = json.loads(archive.read('PUBLIC_EXPORT_MANIFEST.json'))
            included = manifest['included_members']
            expected = {row['member'] for row in included}
            require(set(names) == expected | {'PUBLIC_EXPORT_MANIFEST.json'},
                    f'Unexpected or missing members: {path.name}')
            for row in included:
                payload = archive.read(row['member'])
                require(len(payload) == row['size'], f'Member size mismatch: {row["member"]}')
                require(hashlib.sha256(payload).hexdigest() == row['sha256'],
                        f'Member SHA256 mismatch: {row["member"]}')
            entry = {'archive': path.name, 'verified_members': len(included),
                     'sha256': spec['sha256'], 'crc': 'PASS'}
            if path.name in ('L6_mechanism.zip', 'L8_validation.zip'):
                rows = list(csv.DictReader(io.StringIO(archive.read('metrics.csv').decode('utf-8'))))
                summary = json.loads(archive.read('summary.json'))
                nconditions, nrows = (160, 800) if path.name.startswith('L6') else (80, 480)
                require(summary['status'] == 'PASS' and summary['conditions_completed'] == nconditions,
                        f'Incomplete study: {path.name}')
                require(len(rows) == nrows, f'Wrong record count: {path.name}')
                keys = [(r['step'], r['budget'], r['seed'], r['method']) for r in rows]
                require(len(set(keys)) == len(keys), f'Duplicate result rows: {path.name}')
                conditions = {(r['step'], r['budget'], r['seed']) for r in rows}
                require(len(conditions) == nconditions, f'Condition count mismatch: {path.name}')
                for row in rows:
                    require(int(row['actual_shots']) <= int(row['budget']), 'Shot budget exceeded')
                    for metric in ('nonperiodic_relative_L1', 'periodic_L1'):
                        if row.get(metric):
                            require(math.isfinite(float(row[metric])) and float(row[metric]) >= 0,
                                    f'Invalid error metric: {path.name}')
                entry.update(conditions=nconditions, result_rows=nrows)
                if path.name.startswith('L8'):
                    for (step, budget), printed in EXPECTED_PHYSICAL.items():
                        for method, reference in zip(METHODS, printed):
                            values = [float(r['nonperiodic_relative_L1']) for r in rows
                                      if float(r['step']) == step and int(r['budget']) == budget
                                      and r['method'] == method]
                            require(len(values) == 20, 'L8 cell must contain exactly 20 repeats')
                            median = 100 * statistics.median(values)
                            require(abs(median - reference) <= 0.00051,
                                    f'Manuscript median mismatch: {step}, {budget}, {method}')
                            report['l8_physical_medians_percent'].append(
                                {'step': step, 'budget': budget, 'method': method,
                                 'repeats': 20, 'computed': median, 'printed': reference})
        report['archives'].append(entry)
        print(f'PASS {path.name}: {len(included)} members verified', flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--report', type=Path, help='Optional JSON report destination')
    args = parser.parse_args()
    try:
        report = verify(args.root.expanduser().resolve())
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('PASS: all released archives and the manuscript L8 median table agree.')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
