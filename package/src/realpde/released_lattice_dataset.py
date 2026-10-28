"""Training-only exact decimation phases of released raw PIV; no generated values."""
import h5py,numpy as np,torch
from .data import _field_group
class ReleasedLatticeDataset:
 def __init__(self,cases,stride=1,budget=None):
  self.cases=list(cases);self.flows=[];self.windows=[]
  for ci,c in enumerate(cases):
   assert (c.height,c.width)==(64,128) and not c.has_pressure
   with h5py.File(c.path,'r') as f:
    g=_field_group(f);uv=np.empty((c.frames,64,128,2),np.float32)
    uv[...,0]=g['u'][:];uv[...,1]=g['v'][:]
   t=torch.from_numpy(uv);assert torch.isfinite(t).all();self.flows.append(t)
   self.windows.extend((ci,s) for s in range(0,c.frames-79+1,stride))
   if budget:budget.check()
 def __len__(self):return len(self.windows)
 def phase(self,index,phase):
  assert 0<=phase<4
  ci,start=self.windows[index];dy,dx=divmod(phase,2);uv=self.flows[ci][start:start+40,dy::2,dx::2]
  assert uv.shape==(40,32,64,2)
  native=torch.cat((uv,torch.zeros_like(uv[...,:1])),dim=-1)
  return native[:20],native[20:],torch.tensor(index,dtype=torch.long)
 def __getitem__(self,index):return self.phase(index,0)
 @property
 def storage_bytes(self):return sum(t.numel()*t.element_size() for t in self.flows)
