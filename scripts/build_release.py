"""Build from a clean generated staging tree and verify packaged source/static bytes."""
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

project = Path(__file__).resolve().parents[1]
# setuptools retains obsolete hashed frontend assets in build/lib across releases.
staging = project / 'build'
if staging.exists():
    shutil.rmtree(staging)
subprocess.run(['uv', 'build', '--wheel', '--out-dir', 'dist'], cwd=project, check=True)
wheel = project / 'dist/posttrain_timemachine-0.2.0-py3-none-any.whl'
with zipfile.ZipFile(wheel) as archive:
    source = {str(p.relative_to(project)) for p in (project / 'ptm').rglob('*')
              if p.is_file() and ('__pycache__' not in p.parts) and (p.suffix == '.py' or '/web/' in str(p))}
    packaged = {p for p in archive.namelist() if p.startswith('ptm/') and not p.endswith('/')}
    assert packaged == source, {'missing': sorted(source-packaged), 'unexpected': sorted(packaged-source)}
    assert all(archive.read(p) == (project / p).read_bytes() for p in source)
    result = {'wheel': str(wheel), 'source_files': len(source),
              'static_files': sum(p.startswith('ptm/web/') for p in source),
              'exact_file_set': True, 'all_packaged_bytes_match_source': True}
(project / 'output/release-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
