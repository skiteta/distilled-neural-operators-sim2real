"""Training-only exact temporal subsampling of released measurements."""
import torch
from .temporal_start_dataset import ReleasedTemporalStartDataset
class ReleasedCadenceDataset(ReleasedTemporalStartDataset):
 def can_double(self,index):
  ci,start=self.windows[index]
  return start+79<=len(self.flows[ci])
 def cadence(self,index,cadence):
  assert cadence in (0,1)
  ci,start=self.windows[index];spacing=cadence+1
  assert cadence==0 or self.can_double(index)
  native=self.flows[ci][start:start+40*spacing:spacing]
  assert native.shape==(40,32,64,3)
  return native[:20],native[20:],torch.tensor(index,dtype=torch.long)
 def __getitem__(self,index):return self.cadence(index,0)
