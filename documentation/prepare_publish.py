"""Prepare the rendered site for upload; does not contact or change GitHub."""
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
BUILD = ROOT/'build/html'
DESTINATION = ROOT/'publishing/site'
ARCHIVE = ROOT/'publishing/github-pages-site.zip'

def main():
    if not (BUILD/'index.html').is_file():
        raise SystemExit('Build the HTML site before preparing publication.')
    DESTINATION.mkdir(parents=True,exist_ok=True)
    target = DESTINATION.resolve()
    assert target.is_relative_to(ROOT.resolve()) and target.name == 'site' and target.parent.name == 'publishing'
    shutil.rmtree(target)
    DESTINATION.mkdir(parents=True)
    # Only the canonical rendered HTML tree is copied, not the environment or datasets.
    shutil.copytree(BUILD,DESTINATION,dirs_exist_ok=True)
    (DESTINATION/'.nojekyll').write_text('',encoding='utf-8')
    files = sorted(path for path in BUILD.rglob('*') if path.is_file())
    with ZipFile(ARCHIVE,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path,path.relative_to(BUILD).as_posix())
        archive.writestr('.nojekyll','')
    with ZipFile(ARCHIVE) as archive:
        assert 'index.html' in archive.namelist()
        assert '.nojekyll' in archive.namelist()
        assert archive.testzip() is None
    result = {'site_folder':str(DESTINATION),'zip':str(ARCHIVE),'files':len(files)+1,'archive_bytes':ARCHIVE.stat().st_size,'deployment':'not performed'}
    (ROOT/'publishing/preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
