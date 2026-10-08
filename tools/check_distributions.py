"""Check archive contents and metadata against the intended release source."""
import argparse
import email
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    rb'gh[pousr]_[A-Za-z0-9]{30,}',
    rb'github_pat_[A-Za-z0-9_]{40,}',
    rb'pypi-[A-Za-z0-9_-]{40,}',
    rb'AKIA[0-9A-Z]{16}',
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
]

def check_metadata(data, version):
    message = email.message_from_bytes(data)
    assert message['Name'] == 'bactoscoop'
    assert message['Version'] == version
    assert message['Requires-Python'] == '<3.11,>=3.10'
    description = data.decode('utf-8').replace('\r\n', '\n').split('\n\n', 1)[1]
    assert description.strip() == (ROOT / 'README.md').read_text(encoding='utf-8').strip(), 'Wrong README in distribution metadata'
    assert 'update_docs.cmd' not in description and 'PUBLISHING.md' not in description

def inspect(directory):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    version = config['project']['version']
    code = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in (ROOT / 'bactoscoop').glob('*.py')}
    archives = sorted(directory.glob('bactoscoop-*'))
    assert len(archives) == 2 and {path.suffix for path in archives} == {'.whl', '.gz'}, 'Expected exactly one wheel and one source archive'
    report = {'version': version, 'archives': []}
    for archive_path in archives:
        if archive_path.suffix == '.whl':
            with zipfile.ZipFile(archive_path) as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            metadata = {f'bactoscoop-{version}.dist-info/{name}' for name in
                        ('METADATA', 'WHEEL', 'top_level.txt', 'RECORD', 'licenses/LICENSE')}
            for name in contents:
                assert name in code or name in metadata, f'Unexpected wheel member: {name}'
            check_metadata(contents[f'bactoscoop-{version}.dist-info/METADATA'], version)
        else:
            prefix = f'bactoscoop-{version}/'
            with tarfile.open(archive_path, 'r:gz') as archive:
                assert all(member.isdir() or member.isfile() for member in archive.getmembers()), 'Source archive contains special files'
                assert all(member.name == prefix.rstrip('/') or member.name.startswith(prefix)
                           for member in archive.getmembers()), 'Unexpected source archive prefix'
                contents = {member.name[len(prefix):]: archive.extractfile(member).read()
                            for member in archive.getmembers() if member.isfile() and member.name.startswith(prefix)}
            allowed = {'LICENSE', 'README.md', 'pyproject.toml', 'MANIFEST.in', 'setup.cfg', 'PKG-INFO'}
            for name in contents:
                assert name in code or name in allowed or name.startswith('bactoscoop.egg-info/'), f'Unexpected source member: {name}'
            check_metadata(contents['PKG-INFO'], version)
            for name in ['README.md', 'pyproject.toml', 'MANIFEST.in', 'LICENSE']:
                assert contents[name].replace(b'\r\n', b'\n') == (ROOT / name).read_bytes().replace(b'\r\n', b'\n'), f'Source archive differs: {name}'
        for name, data in code.items():
            assert contents.get(name) == data, f'Incorrect package source: {name}'
        for name, data in contents.items():
            assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
            assert not any(re.search(pattern, data) for pattern in SECRET_PATTERNS), f'Possible secret in archive member: {name}'
            assert not re.search(rb'[A-Za-z]:[/\\]Users[/\\]', data), f'Personal filesystem path in archive member: {name}'
        report['archives'].append({'file': archive_path.name, 'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                                   'members': sorted(contents), 'package_source_files': len(code),
                                   'readme_matches': True, 'secret_pattern_matches': 0})
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = inspect(args.directory.resolve())
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
