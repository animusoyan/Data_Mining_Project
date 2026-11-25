"""Regenerate the archived descriptive notebook exports in a fresh run directory."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from validate_descriptive import validate_exports

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'outputs/experiments/descriptive_recovery'
NOTEBOOKS = ('notebooks/analysis.ipynb', 'notebooks/daily_patterns.ipynb')
CLEAN = 'data/processed/flights_clean.csv'
INPUTS = (*NOTEBOOKS, 'src/__init__.py', 'src/patterns.py', CLEAN)
HISTORICAL_REPORTS = ('eda_findings.md', 'eda_followups.md', 'daily_patterns_findings.md')
PACKAGES = ('pandas', 'numpy', 'scipy', 'scikit-learn', 'joblib', 'matplotlib', 'ipython', 'threadpoolctl', 'pillow')
TABLE_CONTEXT = {
    'monthly': ('Coverage, delay rates and severity by month', 'All flights by scheduled departure month'),
    'weekly': ('Coverage, delay rates and severity by Sunday-ending week', 'All flights including partial weeks'),
    'airlines': ('Coverage, delay rates and severity by carrier', 'All flights by carrier'),
    'airports': ('Coverage, delay rates and severity by NYC origin', 'All flights by origin'),
    'routes': ('Coverage, delay rates and severity by route', 'All flights by origin and destination'),
    'hourly': ('Coverage, delay rates and severity by scheduled hour', 'All flights by NYC local scheduled hour'),
    'daily': ('Coverage, delay rates and severity by date', 'All flights by scheduled departure date'),
    'selected_routes': ('Delay metrics for the ten busiest routes', 'Flights on ten routes ranked by total flights'),
    'weekday_hour': ('Delay counts and rates by weekday and scheduled hour', 'All flights by NYC local schedule'),
    'airport_month': ('Monthly label and delay counts by origin', 'All months and origins, observed labels counted'),
    'carrier_month': ('Monthly label and delay counts by carrier', 'All months and carriers, observed labels counted'),
    'july_top_days': ('Five highest-rate days in July', 'Flights on those dates across the NYC network'),
    'worst_days': ('Highest-rate eligible days with label coverage and delay severity', 'NYC network dates meeting the notebook support rules'),
    'worst_day_origins': ('Highest-rate eligible day by origin versus annual rates', 'Flights on the selected NYC date by origin'),
    'worst_day_carriers': ('Highest-rate eligible day by supported carrier versus annual rates', 'Flights on the selected NYC date for carriers meeting minimum label support'),
    'matched_airline_comparisons': ('Raw and standardized carrier rates on shared schedules', 'Observed labels for two selected carriers per route with supported common month/time bands'),
    'airport_month_time_bands': ('Morning/evening coverage and delay rates', 'Flights scheduled 05:00–09:59 or 17:00–21:59 local time by origin and month'),
    'evening_morning_contrasts': ('Evening minus morning rate in percentage points', 'Origin/month pairs with at least 100 observed labels in each band'),
    'pattern_daily_summary': ('Daily clustering inputs and eligibility', 'All 365 NYC network dates'),
    'cluster_selection': ('Silhouette, inertia and smallest cluster by candidate k', 'Eligible full-network dates for k=2 through k=6'),
    'cluster_stability': ('Adjusted Rand agreement against five additional seeds', 'Eligible full-network dates at the selected k'),
    'cluster_profiles': ('Day-weighted summaries by ordered cluster ID', 'Eligible full-network dates'),
    'daily_patterns': ('Daily assignments, anomaly baselines, scores and nullable flags', 'All 365 NYC network dates'),
    'anomaly_sensitivity': ('Primary and alternative past-only scoring counts', 'Full-network dates with sufficient valid history per rule'),
    'delay_anomalies': ('Dates flagged for unusually high delay rate', 'Flagged eligible NYC network dates'),
    'missing_label_anomalies': ('Dates flagged for unusually high missing-label rate', 'Flagged NYC network dates with sufficient history'),
    'cluster_anomaly_overlap': ('Counts of scorable days and flagged days by cluster', 'Eligible full-network dates with cluster membership'),
    'pattern_case_studies': ('Four historical examples of cluster and flag combinations', 'Four specified NYC network dates, including an ineligible date'),
}
FIGURE_CONTEXT = {
    'temporal_trends': ('Monthly and weekly trends, severity and missing-label coverage', 'All NYC departures'),
    'delay_distributions': ('Departure-delay histogram and monthly boxplots', 'Observed departure delays with display limits'),
    'airline_airport_comparison': ('Delay rates with sample counts by carrier and origin', 'Observed departure labels by carrier and origin'),
    'selected_routes': ('Delay rates and severity on the busiest routes', 'Observed departure delays on the ten busiest routes'),
    'airline_airport_distributions': ('Delay distributions by carrier and origin', 'Observed departure delays with outlier points hidden'),
    'airline_airport_missing_labels': ('Missing departure-label share by carrier and origin', 'All departures by carrier and origin'),
    'weekday_hour_heatmap': ('Delay rates and observed counts by weekday and scheduled hour', 'NYC departures grouped by schedule'),
    'followup_july_daily': ('Calendar-month daily-rate distributions and July trend', 'Full-network dates'),
    'followup_july_groups': ('Monthly origin and major-carrier delay rates', 'Observed labels for all origins and five largest carriers by observed flights'),
    'followup_disrupted_days': ('Highest daily delay rates and origin rates versus annual rates on the top date', 'Full-network dates with the notebook eligibility rule for delay rates'),
    'followup_matched_airlines': ('Raw and standardized within-route airline comparisons', 'Selected carriers and supported common schedules'),
    'followup_evening_within_groups': ('Morning versus evening monthly rates by origin', 'Supported origin/month pairs'),
    'cluster_selection': ('Silhouette and inertia curves', 'Eligible full-network dates across candidate k values'),
    'cluster_timeline': ('Daily delay rates by assigned cluster', 'Eligible full-network dates'),
    'historical_anomalies': ('Daily rates, past-only thresholds and flags', 'Full-network dates with coverage rules applied'),
}

# Execute unmodified Python cells in a subprocess whose entire project root is copied.
EXECUTE_NOTEBOOK = '''
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits

path = Path(sys.argv[1])
notebook = json.loads(path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
plt.show = lambda: None  # Plot exports already occur before interactive display.
with threadpool_limits(limits=1):
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            print(f"Executing {path.name} cell {number}", flush=True)
            source = "".join(cell["source"])
            exec(compile(source, f"{path.name}:cell_{number}", "exec"), namespace)
'''


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).splitlines()


def preservation_inventory(skip=None):
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True)
    paths = {ROOT / name for name in tracked.split('\0') if name}
    for folder in ('data', 'outputs', 'reports', 'notebooks', 'src', 'experiments', 'tests'):
        paths.update(path for path in (ROOT / folder).rglob('*')
                     if path.is_file() and not {'__pycache__', '.ipynb_checkpoints'} & set(path.parts)
                     and path.suffix != '.pyc')
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)
            if path.is_file() and (skip is None or not path.is_relative_to(skip))}


def execute_copies(run, manifest):
    with TemporaryDirectory(prefix='.workspace_', dir=run) as directory:
        workspace = Path(directory)
        for name in INPUTS:
            copy = workspace / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, copy)
            if digest(copy) != manifest['input_hashes'][name]:
                raise ValueError(f'Copied input differs from source: {name}')
        (workspace / 'reports').mkdir()
        (workspace / 'tmp').mkdir()
        environment = {'MPLCONFIGDIR': str(workspace / '.matplotlib'),
                       'XDG_CACHE_HOME': str(workspace / '.cache'),
                       'TMPDIR': str(workspace / 'tmp'), 'OMP_NUM_THREADS': '1'}
        for notebook in NOTEBOOKS:
            log = run / (Path(notebook).stem + '.log')
            command = [sys.executable, '-I', '-B', '-c', EXECUTE_NOTEBOOK, notebook]
            manifest['notebook_executions'].append({'argv': command, 'cwd': str(workspace),
                                                    'environment_overrides': environment, 'log': log.name})
            save_json(run / 'provenance.json', manifest)
            with log.open('x', encoding='utf-8') as stream:
                subprocess.run(command, cwd=workspace, env={**os.environ, **environment}, check=True,
                               stdout=stream, stderr=subprocess.STDOUT)
        for folder in ('tables', 'figures', 'models'):
            shutil.move(workspace / 'outputs' / folder, run / folder)
        shutil.move(workspace / 'reports', run / 'reports')


def index_artifacts(run):
    from PIL import Image

    artifacts = []

    def add(path, purpose, population, schema, **details):
        artifacts.append({'path': path.relative_to(run).as_posix(), 'status': 'regenerated',
                          'purpose': purpose, 'population': population, 'schema': schema,
                          'sha256': digest(path), **details})

    for name, (purpose, population) in TABLE_CONTEXT.items():
        path = run / 'tables' / (name + '.csv')
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            rows = sum(1 for _ in reader)
        schema = {'format': 'CSV', 'columns': columns}
        if name == 'pattern_case_studies':
            schema['unnamed_first_column'] = 'date'
        add(path, purpose, population, schema, rows=rows)
    for name, (purpose, population) in FIGURE_CONTEXT.items():
        path = run / 'figures' / (name + '.png')
        with Image.open(path) as figure:
            if figure.format != 'PNG':
                raise ValueError(f'Expected PNG at {path}')
            width, height = figure.size
            figure.verify()
        add(path, purpose, population, {'format': 'PNG', 'width': width, 'height': height})
    path = run / 'models/daily_patterns_config.json'
    add(path, 'Frozen descriptive parameters and data identity', 'Full-year NYC departures',
        {'format': 'JSON object', 'fields': list(json.loads(path.read_text(encoding='utf-8')))})
    add(run / 'models/daily_clusters.joblib', 'Descriptive standardizer, K-means and ordered cluster IDs',
        'Eligible days across the full NYC network',
        {'format': 'Joblib mapping', 'keys': ['scaler', 'kmeans', 'ordered_cluster_ids', 'features']})
    for name in HISTORICAL_REPORTS:
        add(run / 'reports' / name, 'Notebook-generated evidence for archival comparison',
            'Full-year NYC departures or network dates, with report-specific support conditions',
            {'format': 'Markdown'})
    expected = {item['path'] for item in artifacts}
    actual = {path.relative_to(run).as_posix() for folder in ('tables', 'figures', 'models', 'reports')
              for path in (run / folder).iterdir() if path.is_file()}
    if expected != actual:
        raise ValueError(f'Artifact index differs from exported files: {sorted(expected ^ actual)}')
    return {'scope': 'NYC Flights 2013 full-year description',
            'population_notes': [
                'Delay rates use observed departure-delay labels. Missing values in cluster and anomaly flag columns mean unknown.',
                'Daily clusters and anomaly flags describe the full NYC network. They are not recalculated for filtered airline or airport populations.',
                'Full-year descriptive outcomes must never enter pre-departure classifier features.',
            ], 'artifacts': artifacts}


def recover(run_id, search_note):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Use a simple recovery run ID')
    run = RECOVERY / run_id
    # Check immutability before inspecting or copying source inputs.
    run.mkdir(parents=True, exist_ok=False)
    before = preservation_inventory()
    manifest = {
        'status': 'running',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, '-B', *sys.argv],
        'project_commit': git('rev-parse', 'HEAD')[0],
        'starting_git_status': git('status', '--porcelain=v1', '--untracked-files=all'),
        'interpreter': sys.executable,
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {name: version(name) for name in PACKAGES},
        'input_hashes': {name: before[name] for name in (
            *INPUTS, 'data/raw/flights.csv', 'data/raw/source.json',
            'data/processed/classifier.csv', 'data/processed/missing_departure_delay.csv',
            'scripts/recover_descriptive.py', 'scripts/validate_descriptive.py',
            *(f'reports/{name}' for name in HISTORICAL_REPORTS))},
        'original_export_search': search_note,
        'notebook_executions': [],
    }
    save_json(run / 'preservation_before.json', before)
    save_json(run / 'provenance.json', manifest)
    try:
        execute_copies(run, manifest)
        save_json(run / 'artifact_index.json', index_artifacts(run))
        save_json(run / 'validation.json', validate_exports(ROOT, run, manifest['input_hashes']))
        manifest.update(status='complete', recovered_originals=0, regenerated_artifacts=48,
                        historical_report_copies=3)
    finally:
        after = preservation_inventory(skip=run)
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        save_json(run / 'preservation_check.json', {'files_compared': len(before), 'changed': changed})
        if changed or manifest['status'] != 'complete':
            manifest['status'] = 'failed'
        save_json(run / 'provenance.json', manifest)
        if changed:
            raise RuntimeError(f'Pre-existing project files changed during recovery: {changed}')
    print(f'Regenerated descriptive exports in {run}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, help='fresh directory name under descriptive_recovery')
    parser.add_argument('--search-note', required=True, help='record where original files were sought')
    args = parser.parse_args()
    recover(args.run_id, args.search_note)
