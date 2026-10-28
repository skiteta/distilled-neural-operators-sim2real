"""Inference-only tiled FP32 depthwise5x5 and residual composition.

Uses existing Torch CUDA NVRTC/driver libraries; no filesystem cache, network,
subprocess, learned-weight conversion or process-wide precision changes.
"""
from pathlib import Path
import ctypes as C
import torch
import torch.nn.functional as F
from realpde.fused_bias_ffn_cuda import _load_nvrtc
_SOURCE=r'''
#define BIAS_FIRST __BIAS_FIRST__
#define USE_FMA __USE_FMA__
extern "C" __global__ void depthwise_residual(
 const float* feature,const float* norm,const float* global_value,const float* channel,
 const float* weight,const float* bias,const float* scale,float* output,int channels,int height,int width) {
 __shared__ float tile[12][36];
 __shared__ float weights[25];
 int lane=threadIdx.y*32+threadIdx.x;
 int nc=blockIdx.z,c=nc%channels,base=nc*height*width;
 int bx=blockIdx.x*32,by=blockIdx.y*8;
 for(int t=lane;t<432;t+=256) {
  int yy=by+t/36-2,xx=bx+t%36-2;
  tile[t/36][t%36]=(yy>=0&&yy<height&&xx>=0&&xx<width)?norm[base+yy*width+xx]:0.f;
 }
 if(lane<25)weights[lane]=weight[c*25+lane];
 __syncthreads();
 int y=by+threadIdx.y,x=bx+threadIdx.x;
 if(y>=height||x>=width)return;
 float value=BIAS_FIRST?bias[c]:0.f;
 #pragma unroll
 for(int ky=0;ky<5;++ky) {
  #pragma unroll
  for(int kx=0;kx<5;++kx) {
   float a=tile[threadIdx.y+ky][threadIdx.x+kx],w=weights[ky*5+kx];
   value=USE_FMA?__fmaf_rn(a,w,value):__fadd_rn(__fmul_rn(a,w),value);
  }
 }
 if(!BIAS_FIRST)value=__fadd_rn(value,bias[c]);
 int i=base+y*width+x;
 float sum=__fadd_rn(__fadd_rn(value,global_value[i]),channel[i]);
 output[i]=__fadd_rn(feature[i],__fmul_rn(scale[c],sum));
}
'''
class FusedLocalResidual:
 def __init__(self,device,*,bias_first=True,use_fma=True):
  self.device=torch.device(device);self.bias_first=bool(bias_first);self.use_fma=bool(use_fma)
  if self.device.type!='cuda':return
  nvrtc=_load_nvrtc();self.driver=C.CDLL('libcuda.so.1');P=C.c_void_p
  def bind(lib,name,args):
   f=getattr(lib,name);f.restype=C.c_int;f.argtypes=args;return f
  create=bind(nvrtc,'nvrtcCreateProgram',[C.POINTER(P),C.c_char_p,C.c_char_p,C.c_int,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)])
  compile_program=bind(nvrtc,'nvrtcCompileProgram',[P,C.c_int,C.POINTER(C.c_char_p)])
  log_size=bind(nvrtc,'nvrtcGetProgramLogSize',[P,C.POINTER(C.c_size_t)]);log_get=bind(nvrtc,'nvrtcGetProgramLog',[P,C.c_char_p])
  ptx_size=bind(nvrtc,'nvrtcGetPTXSize',[P,C.POINTER(C.c_size_t)]);ptx_get=bind(nvrtc,'nvrtcGetPTX',[P,C.c_char_p]);destroy=bind(nvrtc,'nvrtcDestroyProgram',[C.POINTER(P)])
  program=P();source=_SOURCE.replace('__BIAS_FIRST__',str(int(self.bias_first))).replace('__USE_FMA__',str(int(self.use_fma)))
  self._check(create(C.byref(program),source.encode(),b'local_residual.cu',0,None,None),'nvrtcCreateProgram')
  try:
   major,minor=torch.cuda.get_device_capability(self.device);opts=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++14',b'--fmad=false'];options=(C.c_char_p*len(opts))(*opts);code=compile_program(program,len(opts),options)
   if code:
    size=C.c_size_t();self._check(log_size(program,C.byref(size)),'log size');log=C.create_string_buffer(size.value);self._check(log_get(program,log),'compile log');raise RuntimeError(log.value.decode())
   size=C.c_size_t();self._check(ptx_size(program,C.byref(size)),'PTX size');self.ptx=C.create_string_buffer(size.value);self._check(ptx_get(program,self.ptx),'PTX get')
  finally:self._check(destroy(C.byref(program)),'destroy compiler')
  load=bind(self.driver,'cuModuleLoadDataEx',[C.POINTER(P),P,C.c_uint,P,P]);get=bind(self.driver,'cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p]);self.launch=bind(self.driver,'cuLaunchKernel',[P,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,P,C.POINTER(P),P]);self.module=P();self.function=P();self._check(load(C.byref(self.module),C.cast(self.ptx,P),0,None,None),'load PTX');self._check(get(C.byref(self.function),self.module,b'depthwise_residual'),'kernel function')
 @staticmethod
 def _check(code,name):
  if code:raise RuntimeError(f'{name} CUDA/NVRTC status {code}')
 def __call__(self,feature,norm,global_value,channel,weight,bias,scale):
  if self.device.type!='cuda':return feature+scale*(F.conv2d(norm,weight,bias,padding=2,groups=norm.shape[1])+global_value+channel)
  if feature.ndim!=4 or feature.dtype!=torch.float32 or not feature.is_cuda:raise ValueError('Expected NCHW CUDA FP32')
  n,c,h,w=feature.shape
  if weight.shape!=(c,1,5,5) or bias.shape!=(c,) or scale.shape!=(1,c,1,1):raise ValueError('Depthwise parameter shape mismatch')
  if n*c>=65536 or min(n,c,h,w)<=0:raise ValueError('Unsupported geometry')
  for t in [feature,norm,global_value,channel,weight,bias,scale]:
   if t.device!=feature.device or t.dtype!=torch.float32 or not t.is_contiguous():raise ValueError('Expected contiguous same-device FP32 arrays')
  if any(t.shape!=feature.shape for t in [norm,global_value,channel]):raise ValueError('Branch shape mismatch')
  output=torch.empty_like(feature);args=[C.c_void_p(t.data_ptr()) for t in [feature,norm,global_value,channel,weight,bias,scale,output]]+[C.c_int(c),C.c_int(h),C.c_int(w)];params=(C.c_void_p*len(args))(*[C.cast(C.byref(v),C.c_void_p).value for v in args]);stream=C.c_void_p(torch.cuda.current_stream(feature.device).cuda_stream)
  self._check(self.launch(self.function,(w+31)//32,(h+7)//8,n*c,32,8,1,0,stream,params,None),'launch tiled depthwise residual');return output
