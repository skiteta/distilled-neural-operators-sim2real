"""Inference-only fixed FFN layout, bias+GLU+GELU and bias+transpose on the current CUDA stream.

Uses CUDA libraries already shipped with torch. Source/PTX stays in memory;
no subprocesses, filesystem writes, environment changes or numerical fitting.
"""
from pathlib import Path
import sys
import ctypes as C
import torch
import torch.nn.functional as F

_SOURCE = r'''
#define WIDTH __WIDTH__
#define HIDDEN __HIDDEN__
#define SPATIAL 2048
extern "C" __global__ void bias_glu_gelu(const float* input, const float* bias, float* output, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * HIDDEN * SPATIAL) return;
    int pixel = i / HIDDEN;
    int c = i % HIDDEN;
    float a = input[pixel * (2 * HIDDEN) + c] + bias[c];
    float b = input[pixel * (2 * HIDDEN) + HIDDEN + c] + bias[HIDDEN + c];
    float value = a * (1.0f / (1.0f + expf(-b)));
    output[i] = (value * 0.5f) * (1.0f + erff(value * 0.70710678118654752440f));
}
extern "C" __global__ void bias_transpose(const float* input, const float* bias, float* output, int batch) {
    __shared__ float tile[32][33];
    int n = blockIdx.z;
    int c = blockIdx.x * 32 + threadIdx.x;
    int pixel = blockIdx.y * 32 + threadIdx.y;
    for (int j=0; j<32; j+=8) {
        if (n<batch && c<WIDTH && pixel+j<SPATIAL)
            tile[threadIdx.y+j][threadIdx.x] = input[(n*SPATIAL+pixel+j)*WIDTH+c] + bias[c];
    }
    __syncthreads();
    int out_c = blockIdx.x * 32 + threadIdx.y;
    int out_pixel = blockIdx.y * 32 + threadIdx.x;
    for (int j=0; j<32; j+=8) {
        if (n<batch && out_c+j<WIDTH && out_pixel<SPATIAL)
            output[(n*WIDTH+out_c+j)*SPATIAL+out_pixel] = tile[threadIdx.x][threadIdx.y+j];
    }
}
'''

def _nvrtc_candidates(torch_file=None, prefix=None):
    """Known CUDA12 runtime locations for pip and conda installations."""
    torch_dir = Path(torch.__file__ if torch_file is None else torch_file).resolve().parent
    install_prefix = Path(sys.prefix if prefix is None else prefix)
    return (
        str(torch_dir.parent / 'nvidia/cuda_nvrtc/lib/libnvrtc.so.12'),
        str(install_prefix / 'lib/libnvrtc.so.12'),
        str(torch_dir / 'lib/libnvrtc.so.12'),
        'libnvrtc.so.12',
    )


def _load_nvrtc(torch_file=None, prefix=None):
    """Load an existing runtime only; never install, spawn a process, or mutate paths."""
    failures = []
    for candidate in _nvrtc_candidates(torch_file, prefix):
        try:
            library = C.CDLL(candidate)
        except OSError as exc:
            failures.append(f'{candidate}: {exc}')
            continue
        version = library.nvrtcVersion
        version.restype = C.c_int
        version.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
        major, minor = C.c_int(), C.c_int()
        code = version(C.byref(major), C.byref(minor))
        if code != 0 or major.value != 12:
            failures.append(f'{candidate}: incompatible NVRTC {major.value}.{minor.value}, status {code}')
            continue
        return library
    raise RuntimeError('An existing CUDA12 NVRTC library is required. Tried: ' + '; '.join(failures))


class FusedBiasFfn:
    def __init__(self, device, width=128, hidden=384):
        self.width, self.hidden = int(width), int(hidden)
        if self.width<=0 or self.hidden<=0 or self.width%32 or self.hidden%32:
            raise ValueError("FFN dimensions must be positive multiples of32")
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            return
        self.nvrtc = _load_nvrtc()
        self.driver = C.CDLL('libcuda.so.1')
        P = C.c_void_p
        def bind(lib, name, args):
            fn = getattr(lib, name); fn.restype = C.c_int; fn.argtypes = args; return fn
        create = bind(self.nvrtc, 'nvrtcCreateProgram', [C.POINTER(P), C.c_char_p, C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)])
        compile_program = bind(self.nvrtc, 'nvrtcCompileProgram', [P, C.c_int, C.POINTER(C.c_char_p)])
        log_size = bind(self.nvrtc, 'nvrtcGetProgramLogSize', [P, C.POINTER(C.c_size_t)])
        log_get = bind(self.nvrtc, 'nvrtcGetProgramLog', [P, C.c_char_p])
        ptx_size = bind(self.nvrtc, 'nvrtcGetPTXSize', [P, C.POINTER(C.c_size_t)])
        ptx_get = bind(self.nvrtc, 'nvrtcGetPTX', [P, C.c_char_p])
        destroy = bind(self.nvrtc, 'nvrtcDestroyProgram', [C.POINTER(P)])
        program = P()
        self._check(create(C.byref(program), _SOURCE.replace('__WIDTH__',str(self.width)).replace('__HIDDEN__',str(self.hidden)).encode(), b'fixed_glu_gelu.cu', 0, None, None), 'nvrtcCreateProgram')
        try:
            major, minor = torch.cuda.get_device_capability(self.device)
            self.options = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++14', b'--fmad=false']
            options = (C.c_char_p * len(self.options))(*self.options)
            code = compile_program(program, len(self.options), options)
            if code:
                size = C.c_size_t(); self._check(log_size(program, C.byref(size)), 'nvrtcGetProgramLogSize')
                log = C.create_string_buffer(size.value); self._check(log_get(program, log), 'nvrtcGetProgramLog')
                raise RuntimeError(f'NVRTC compilation failed {code}: {log.value.decode()}')
            size = C.c_size_t(); self._check(ptx_size(program, C.byref(size)), 'nvrtcGetPTXSize')
            self.ptx = C.create_string_buffer(size.value); self._check(ptx_get(program, self.ptx), 'nvrtcGetPTX')
        finally:
            self._check(destroy(C.byref(program)), 'nvrtcDestroyProgram')
        load = bind(self.driver, 'cuModuleLoadDataEx', [C.POINTER(P), P, C.c_uint, P, P])
        get = bind(self.driver, 'cuModuleGetFunction', [C.POINTER(P), P, C.c_char_p])
        self.launch = bind(self.driver, 'cuLaunchKernel', [P, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, P, C.POINTER(P), P])
        self.module, self.activation_function, self.output_function = P(), P(), P()
        self._check(load(C.byref(self.module), C.cast(self.ptx, P), 0, None, None), 'cuModuleLoadDataEx')
        self._check(get(C.byref(self.activation_function), self.module, b'bias_glu_gelu'), 'cuModuleGetFunction')
        self._check(get(C.byref(self.output_function), self.module, b'bias_transpose'), 'cuModuleGetFunction')

    @staticmethod
    def _check(code, operation):
        if code != 0:
            raise RuntimeError(f'{operation} failed with CUDA/NVRTC status {code}')


    def _launch(self, function, grid, block, value, bias, output):
        args=[C.c_void_p(value.data_ptr()),C.c_void_p(bias.data_ptr()),C.c_void_p(output.data_ptr()),C.c_int(value.shape[0])]
        params=(C.c_void_p*4)(*[C.cast(C.byref(v),C.c_void_p).value for v in args])
        stream=C.c_void_p(torch.cuda.current_stream(value.device).cuda_stream)
        self._check(self.launch(function,*grid,*block,0,stream,params,None),'cuLaunchKernel')

    def __call__(self, value, weight1, bias1, weight2, bias2):
        if self.device.type!='cuda':
            return F.conv2d(F.gelu(F.glu(F.conv2d(value,weight1,bias1),dim=1)),weight2,bias2)
        if value.ndim!=4 or value.shape[1:]!=(self.width,32,64) or value.dtype!=torch.float32 or not value.is_cuda:
            raise ValueError('fixedFFN expects CUDA FP32 N,width,32,64')
        if value.device!=torch.device('cuda',torch.cuda.current_device()):raise ValueError('current CUDA device mismatch')
        batch=value.shape[0]
        if batch<=0 or batch*self.hidden*2048>=2**30:raise ValueError('unsupported batchsize')
        if weight1.shape!=(2*self.hidden,self.width,1,1) or weight2.shape!=(self.width,self.hidden,1,1) or bias1.shape!=(2*self.hidden,) or bias2.shape!=(self.width,):raise ValueError('FFN parameter shape mismatch')
        for tensor in (weight1,weight2,bias1,bias2):
            if tensor.device!=value.device or tensor.dtype!=torch.float32 or not tensor.is_contiguous():raise ValueError('FFN parameters must be contiguous same-device FP32')
        raw=F.conv2d(value.contiguous(memory_format=torch.channels_last),weight1,None)
        if not raw.is_contiguous(memory_format=torch.channels_last):raise RuntimeError('firstconvolution layout changed')
        middle=torch.empty((batch,self.hidden,32,64),device=value.device,dtype=value.dtype,memory_format=torch.channels_last)
        self._launch(self.activation_function,((middle.numel()+255)//256,1,1),(256,1,1),raw,bias1,middle)
        last=F.conv2d(middle,weight2,None)
        if not last.is_contiguous(memory_format=torch.channels_last):raise RuntimeError('secondconvolution layout changed')
        output=torch.empty((batch,self.width,32,64),device=value.device,dtype=value.dtype)
        self._launch(self.output_function,((self.width+31)//32,64,batch),(32,8,1),last,bias2,output)
        return output
