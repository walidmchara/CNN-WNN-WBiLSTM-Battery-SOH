# Colab — Reconstruct the paper's CNN–WNN–WBiLSTM–AM

Original sources, three distinct NASA temperature groups, paper Table 2 architecture, RMSprop 0.001, **58 epochs** and batch size 32. This is a source-grounded reconstruction, not the missing original Keras source.

## Run all three original NASA groups

Add `GithubToken` to Colab Secrets and run the following cell. It uses the original twelve-cell NASA ZIP already cached on Drive, or retrieves and checksum-verifies the official source if needed. A new run folder is created for each temperature group.

```python
import os, sys, json, tempfile, subprocess
from pathlib import Path
from google.colab import drive, userdata
from IPython.display import display, Image

drive.mount('/content/drive')
repo_name = 'walidmchara/CNN-WNN-WBiLSTM-Battery-SOH'
base = Path('/content/drive/MyDrive/Automatika_2025_Reproduction')
source = Path('/content/drive/MyDrive/NASA_original_source')
mat_dir = source / 'mat'
archive = source / '5.+Battery+Data+Set.zip'
results, checkpoints, cache = base / 'results', base / 'checkpoints', base / 'cache'
for directory in [source, mat_dir, results, checkpoints, cache]:
    directory.mkdir(parents=True, exist_ok=True)

secret = userdata.get('GithubToken')
assert secret, 'Colab Secret GithubToken is missing'
repo = Path(tempfile.mkdtemp(prefix='paper_reconstruction_', dir='/content')) / 'repo'
with tempfile.TemporaryDirectory() as auth_dir:
    askpass = Path(auth_dir) / 'askpass.py'
    askpass.write_text(
        '#!/usr/bin/env python3\n'
        "import os,sys\n"
        "print('x-access-token' if 'username' in sys.argv[1].lower() else os.environ['GITHUB_TOKEN'])\n"
    )
    askpass.chmod(0o700)
    auth = os.environ.copy()
    auth.update(GITHUB_TOKEN=secret, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0')
    cloned = subprocess.run(['git','-c','credential.helper=','clone','--depth','1',
      f'https://github.com/{repo_name}.git',str(repo)],
      env=auth, capture_output=True, text=True)
    if cloned.returncode:
        raise RuntimeError('Private GitHub clone failed; check token permissions')
del secret, auth
print('Reconstruction source cloned')

subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(repo/'requirements.txt')],check=True)
subprocess.run([sys.executable,'-u','-m','simple_soh.restore_nasa_mat',
    '--archive',str(archive),'--dest',str(mat_dir),
    '--manifest-out',str(source/'mat12_source_audit.json')],cwd=repo,check=True)

for group in ('nasa_24c','nasa_43c','nasa_4c'):
    run_id = f'automatika2025_{group}_p64_e58_seed42'
    command = [sys.executable,'-u','-m','paper_reproduction.run',
      '--group',group,'--source-root',str(mat_dir),
      '--cache-root',str(cache),'--output-root',str(results),
      '--checkpoint-root',str(checkpoints),'--points','64',
      '--epochs','58','--batch-size','32','--seed','42','--device','auto']
    if (checkpoints/run_id/'latest.pt').exists():
        command.append('--resume')
    print('Starting or resuming:',group,flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2')
    proc = subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in proc.stdout:
        print(line,end='',flush=True)
    if proc.wait():
        raise RuntimeError(f'Run failed for {group}')
    folder = results/run_id
    report = json.loads((folder/'metrics.json').read_text())
    assert report['epochs_completed']==58
    print('Measured metrics:',report['new_run_test'])
    print('Published Table 3:',report['paper_published_reference'])
    for chart in ('soh_prediction.png','convergence.png'):
        display(Image(filename=str(folder/chart)))

print('Three independent NASA paper reconstructions completed on Drive:',results)
```

All snapshots and final weights stay on Google Drive. Rerun this same cell after Colab interruption; the same run resumes from its last completed epoch. The experiment reports **new observed metrics**; do not present them as verified paper reproduction solely because Table 2 settings are matched.

## CALCE CS2-38 — original Excel archives

Keep the four original supplied archives `CS2_35.zip`, `CS2_36.zip`, `CS2_37.zip`, `CS2_38.zip` in `MyDrive/CALCE_original_source/`. They are not automatically transferred from the ChatGPT file library to Colab. Run the following after the NASA cell:

```python
calce_dir = Path('/content/drive/MyDrive/CALCE_original_source')
missing = [name for name in ('CS2_35','CS2_36','CS2_37','CS2_38')
           if not (calce_dir/f'{name}.zip').exists()]
if missing:
    raise FileNotFoundError('Original CALCE archives missing: '+str(missing))
run_id = 'automatika2025_calce_cs2_p64_e58_seed42'
command = [sys.executable,'-u','-m','paper_reproduction.run',
  '--group','calce_cs2','--source-root',str(calce_dir),
  '--cache-root',str(cache),'--output-root',str(results),
  '--checkpoint-root',str(checkpoints),'--epochs','58','--points','64']
if (checkpoints/run_id/'latest.pt').exists():
    command.append('--resume')
subprocess.run(command,cwd=repo,check=True)
report = json.loads((results/run_id/'metrics.json').read_text())
print(report['new_run_test'],report['paper_published_reference'])
```

The CALCE reader uses original source workbook measurement rows, not the derived cycle summaries. The source channels lack temperature, which is zero-filled and declared. No source archive or model weights are uploaded to GitHub.

[Methodology and Table 2 fidelity notes](README.md).
