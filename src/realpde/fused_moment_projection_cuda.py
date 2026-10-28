"""Inference-only FP64 trajectory projection, fused without changing its formula.

Compilation uses the CUDA 12 NVRTC already available to the baseline submission.
The kernel owns no input-dependent state and launches on PyTorch's current stream.
"""
import ctypes as C
import torch
from .portable_bias_ffn_cuda import _load_nvrtc

_SOURCE = r'''
extern "C" __global__ void project_moments(
    const float* a, const float* b, float* out, int batch,
    long long an, long long at, long long ah, long long aw, long long ac,
    long long bn, long long bt, long long bh, long long bw, long long bc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * 2048) return;
    int n = i / 2048, pixel = i % 2048, h = pixel / 64, w = pixel % 64;
    long long ai = n*an + h*ah + w*aw, bi = n*bn + h*bh + w*bw;
    double am0=0., am1=0., bm0=0., bm1=0.;
    for (int t=0; t<20; ++t) {
        am0 += (double)a[ai+t*at]; am1 += (double)a[ai+t*at+ac];
        bm0 += (double)b[bi+t*bt]; bm1 += (double)b[bi+t*bt+bc];
    }
    am0/=20.; am1/=20.; bm0/=20.; bm1/=20.;
    double ea=0., eb=0.;
    for (int t=0; t<20; ++t) {
        double a0=(double)a[ai+t*at]-am0, a1=(double)a[ai+t*at+ac]-am1;
        double b0=(double)b[bi+t*bt]-bm0, b1=(double)b[bi+t*bt+bc]-bm1;
        ea += a0*a0 + a1*a1; eb += b0*b0 + b1*b1;
    }
    ea/=20.; eb/=20.;
    double scale=sqrt(fmax(eb,1e-30)/fmax(ea,1e-30));
    bool live=ea>1e-30 && eb>1e-30;
    for (int t=0; t<20; ++t) {
        long long oi=((long long)n*20*2048+t*2048+pixel)*2;
        double a0=(double)a[ai+t*at]-am0, a1=(double)a[ai+t*at+ac]-am1;
        double b0=(double)b[bi+t*bt]-bm0, b1=(double)b[bi+t*bt+bc]-bm1;
        out[oi]=(float)(bm0+(live?a0*scale:b0));
        out[oi+1]=(float)(bm1+(live?a1*scale:b1));
    }
}
'''

class FusedMomentProjection:
    """Fixed 20 x 32 x 64 UV projection; FP32 storage and FP64 arithmetic."""
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('This inference implementation requires CUDA')
        with torch.cuda.device(self.device):
            torch.cuda.init()
            # Materialize the primary context before CUDA Driver API module load.
            torch.empty(1, device=self.device)
            self.nvrtc, self.driver = _load_nvrtc(), C.CDLL('libcuda.so.1')
            P=C.c_void_p
            def bind(lib,name,args):
                fn=getattr(lib,name); fn.restype=C.c_int; fn.argtypes=args; return fn
            create=bind(self.nvrtc,'nvrtcCreateProgram',[C.POINTER(P),C.c_char_p,C.c_char_p,C.c_int,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)])
            compile_=bind(self.nvrtc,'nvrtcCompileProgram',[P,C.c_int,C.POINTER(C.c_char_p)])
            getlogsize=bind(self.nvrtc,'nvrtcGetProgramLogSize',[P,C.POINTER(C.c_size_t)])
            getlog=bind(self.nvrtc,'nvrtcGetProgramLog',[P,C.c_char_p])
            getsize=bind(self.nvrtc,'nvrtcGetPTXSize',[P,C.POINTER(C.c_size_t)])
            getptx=bind(self.nvrtc,'nvrtcGetPTX',[P,C.c_char_p])
            destroy=bind(self.nvrtc,'nvrtcDestroyProgram',[C.POINTER(P)])
            program=P()
            self._check(create(C.byref(program),_SOURCE.encode(),b'project_moments.cu',0,None,None),'create')
            try:
                major,minor=torch.cuda.get_device_capability(self.device)
                self.options=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++14',b'--fmad=false']
                options=(C.c_char_p*len(self.options))(*self.options)
                code=compile_(program,len(options),options)
                if code:
                    size=C.c_size_t(); self._check(getlogsize(program,C.byref(size)),'log size')
                    log=C.create_string_buffer(size.value); self._check(getlog(program,log),'log')
                    raise RuntimeError(f'NVRTC error {code}: {log.value.decode()}')
                size=C.c_size_t(); self._check(getsize(program,C.byref(size)),'PTX size')
                self.ptx=C.create_string_buffer(size.value); self._check(getptx(program,self.ptx),'PTX')
            finally:
                self._check(destroy(C.byref(program)),'destroy program')
            load=bind(self.driver,'cuModuleLoadDataEx',[C.POINTER(P),P,C.c_uint,P,P])
            getfn=bind(self.driver,'cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p])
            self.launch=bind(self.driver,'cuLaunchKernel',[P,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,P,C.POINTER(P),P])
            self.module,self.function=P(),P()
            self._check(load(C.byref(self.module),C.cast(self.ptx,P),0,None,None),'load module')
            self._check(getfn(C.byref(self.function),self.module,b'project_moments'),'get function')

    @staticmethod
    def _check(code,operation):
        if code: raise RuntimeError(f'{operation}: CUDA/NVRTC status {code}')

    def __call__(self,candidate,baseline):
        for x in (candidate,baseline):
            if x.dtype!=torch.float32 or not x.is_cuda or x.shape[1:]!=(20,32,64,2):
                raise ValueError('Expected CUDA FP32 [N,20,32,64,2]')
            if x.requires_grad and torch.is_grad_enabled():
                raise RuntimeError('FusedMomentProjection is inference only')
        if candidate.shape!=baseline.shape or candidate.device!=baseline.device:
            raise ValueError('Projection inputs must have identical shapes and devices')
        if candidate.device!=torch.device('cuda',torch.cuda.current_device()):
            raise ValueError('Current CUDA device mismatch')
        if len(candidate)>2**20: raise ValueError('Unsupported batch size')
        out=torch.empty(candidate.shape,device=candidate.device,dtype=candidate.dtype)
        if not len(candidate): return out
        args=[C.c_void_p(candidate.data_ptr()),C.c_void_p(baseline.data_ptr()),C.c_void_p(out.data_ptr()),C.c_int(len(candidate))]
        args += [C.c_longlong(s) for x in (candidate,baseline) for s in x.stride()]
        params=(C.c_void_p*len(args))(*[C.cast(C.byref(v),C.c_void_p).value for v in args])
        stream=C.c_void_p(torch.cuda.current_stream(candidate.device).cuda_stream)
        self._check(self.launch(self.function,(len(candidate)*2048+127)//128,1,1,128,1,1,0,stream,params,None),'launch')
        return out
