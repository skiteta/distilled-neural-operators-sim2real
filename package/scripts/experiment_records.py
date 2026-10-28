"""One writer for new experiment events through the mandatory deployment guard."""
from pathlib import Path
import fcntl,hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def register(e,path):
 path=Path(path);assert not path.exists()
 with (ROOT/'.records/release_event_writer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  w=read(ROOT/'.records/work.json');e['id']='E'+str(max(int(v['id'][1:]) for v in w['events'] if re.fullmatch(r'E\d+',v['id']))+1).zfill(6);write(path,e)
  r=subprocess.run(['python3',str(ROOT/'scripts/work_release.py'),'record','--event',rel(path)],cwd=ROOT,capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stdout+r.stderr)
  ledger=ROOT/'EXPERIMENTS.jsonl';item={k:e[k] for k in ['id','candidate','kind','summary']};item.update(event_id=e['id'],event_ref=rel(path),event_sha256=sha(path))
  with ledger.open('ab+') as f:
   f.seek(0,2);size=f.tell();f.seek(max(0,size-1));last=f.read(1);f.seek(0,2);f.write((b'' if not size or last==b'\n' else b'\n')+json.dumps(item,ensure_ascii=False).encode()+b'\n')
  result=json.loads(r.stdout);print(json.dumps({'event':e['id'],'assessment':result['assessment']['label']},ensure_ascii=False),flush=True);return e['id']
