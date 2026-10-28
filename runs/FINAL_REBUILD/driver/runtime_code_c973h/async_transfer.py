"""Per-call owned pinned outputs, bounded48-row GPU staging, independent copy stream."""
import numpy as np
import torch

def install(S):
 @torch.inference_mode()
 def predict(array,include_center):
  keys=['prediction','lower','upper']+(['adopted_center'] if include_center else [])
  host={key:torch.empty((len(array),*array.shape[1:]),dtype=torch.float32,pin_memory=True) for key in keys}
  copy_stream=torch.cuda.Stream(device=S._DEVICE)
  with S._GRAPH_LOCK:
   current=torch.cuda.current_stream(S._DEVICE)
   for group_start in range(0,len(array),S._TRANSFER_GROUP_SIZE):
    chunk=np.ascontiguousarray(array[group_start:group_start+S._TRANSFER_GROUP_SIZE]);valid=len(chunk);padded=((valid+S._INTERNAL_BATCH_SIZE-1)//S._INTERNAL_BATCH_SIZE)*S._INTERNAL_BATCH_SIZE
    if padded!=valid:
     p=np.empty((padded,*chunk.shape[1:]),dtype=np.float32);p[:valid]=chunk;p[valid:]=chunk[-1:];chunk=p
    batch=torch.from_numpy(chunk).to(S._DEVICE)
    if S._GRAPH_LANE is None:S._GRAPH_LANE=S._GraphLane(batch[:S._INTERNAL_BATCH_SIZE])
    lane=S._GRAPH_LANE;lane.stream.wait_stream(current)
    with torch.cuda.stream(lane.stream):
     outputs=[torch.empty((valid,*v.shape[1:]),dtype=v.dtype,device=v.device) for v in lane.values[:len(keys)]]
     for start in range(0,padded,S._INTERNAL_BATCH_SIZE):
      count=min(S._INTERNAL_BATCH_SIZE,valid-start);lane.input.copy_(batch[start:start+S._INTERNAL_BATCH_SIZE]);lane.graph.replay()
      for dst,val in zip(outputs,lane.values):dst[start:start+count].copy_(val[:count])
     batch.record_stream(lane.stream)
    copy_stream.wait_stream(lane.stream)
    with torch.cuda.stream(copy_stream):
     for key,val in zip(keys,outputs):
      host[key][group_start:group_start+valid].copy_(val,non_blocking=True);val.record_stream(copy_stream)
    # Preserve bounded live GPU staging before more than two groups accumulate.
    if group_start and (group_start//S._TRANSFER_GROUP_SIZE)%2==1:copy_stream.synchronize()
   copy_stream.synchronize()
   with torch.cuda.stream(lane.stream):
    lane.input.zero_()
    for val in lane.values:val.zero_()
   current.wait_stream(lane.stream)
  result={k:v.numpy() for k,v in host.items()}
  if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('Nonfinite output')
  if not np.all(result['lower']<=result['upper']):raise RuntimeError('Invalid interval')
  return result
 S._cuda_predict=predict
