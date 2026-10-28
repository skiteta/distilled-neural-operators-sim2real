"""Mirror of released_lattice_cycle.package() for the LATCAD candidate arm (midpoint/final EMA 50:50)."""
import io,sys,json,hashlib,zipfile,torch
from pathlib import Path
R=Path(__file__).resolve().parents[2]
WEIGHTS={'model.pt':'point','center_head.pt':'center','radius_head.pt':'radius','hidden_head.pt':'hidden','observed_center_head.pt':'observed'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def average(a,b):
    assert a.keys()==b.keys();return {k:(a[k]*.5+b[k]*.5 if a[k].is_floating_point() or a[k].is_complex() else a[k].clone()) for k in a}
def build(train_dir,plan_path,target,recipe=None):
    m=json.loads(Path(plan_path).read_text());n=m['training']['steps'];mid=m['middle_step']
    t=Path(train_dir);end=torch.load(t/f'resume_{n}.pt',map_location='cpu',weights_only=False);middle=torch.load(t/f'resume_{mid}.pt',map_location='cpu',weights_only=False)
    fp=average(middle['ema_point'],end['ema_point']);fh=average(middle['ema_heads'],end['ema_heads'])
    states={name:(fp if key=='point' else {k[len(key)+1:]:v for k,v in fh.items() if k.startswith(key+'.')}) for name,key in WEIGHTS.items()}
    teacher=R/m['teacher_archive']['path'];runtime=R/m['runtime_baseline']['path'];target=Path(target);assert not target.exists()
    with zipfile.ZipFile(teacher) as origin,zipfile.ZipFile(runtime) as rt,zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name in rt.namelist():
            data=rt.read(name)
            if name in WEIGHTS:
                payload=torch.load(io.BytesIO(origin.read(name)),map_location='cpu',weights_only=False);payload['state_dict']=states[name];out=io.BytesIO();torch.save(payload,out);data=out.getvalue()
            z.writestr(name,data)
        for rname,rbytes in (recipe or {}).items():   # non-executed provenance for organizer retraining
            assert rname.startswith('training_recipe/') and rname not in rt.namelist();z.writestr(rname,rbytes)
    return states
if __name__=='__main__':
    build(sys.argv[1],sys.argv[2],sys.argv[3]);print('built',sys.argv[3],sha(sys.argv[3]))
