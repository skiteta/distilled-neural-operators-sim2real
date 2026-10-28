"""Inference-only FP32 GLU(dim=1)+exact GELU on the current CUDA stream.

Uses CUDA libraries already shipped with torch. Source/PTX stays in memory;
no subprocesses, filesystem writes, environment changes or numerical fitting.
"""
from pathlib import Path
import ctypes as C
import torch
import torch.nn.functional as F

_SOURCE = r'''
extern "C" __global__ void glu_gelu(const float* input, float* output, int batch, int channels, int spatial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch * channels * spatial;
    if (i >= size) return;
    int n = i / (channels * spatial);
    int q = i - n * channels * spatial;
    float a = input[n * (2 * channels * spatial) + q];
    float b = input[n * (2 * channels * spatial) + channels * spatial + q];
    float gate = 1.0f / (1.0f + expf(-b));
    float value = a * gate;
    output[i] = (value * 0.5f) * (1.0f + erff(value * 0.70710678118654752440f));
}
'''

class FusedGluGelu:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            return
        root = Path(torch.__file__).resolve().parent.parent / 'nvidia/cuda_nvrtc/lib'
        library = root / 'libnvrtc.so.12'
        if not library.is_file():
            raise RuntimeError('The existing torch CUDA12 NVRTC library is required')
        self.nvrtc = C.CDLL(str(library))
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
        self._check(create(C.byref(program), _SOURCE.encode(), b'fixed_glu_gelu.cu', 0, None, None), 'nvrtcCreateProgram')
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
        self.module, self.function = P(), P()
        self._check(load(C.byref(self.module), C.cast(self.ptx, P), 0, None, None), 'cuModuleLoadDataEx')
        self._check(get(C.byref(self.function), self.module, b'glu_gelu'), 'cuModuleGetFunction')

    @staticmethod
    def _check(code, operation):
        if code != 0:
            raise RuntimeError(f'{operation} failed with CUDA/NVRTC status {code}')

    def __call__(self, value):
        if self.device.type != 'cuda':
            return F.gelu(F.glu(value, dim=1))
        if value.ndim != 4 or value.dtype != torch.float32 or not value.is_cuda or not value.is_contiguous() or value.shape[1] % 2:
            raise ValueError('fused GLU-GELU expects contiguous CUDA FP32 NCHW with even channels')
        if value.device != torch.device('cuda', torch.cuda.current_device()):
            raise ValueError('input must be on the current CUDA device')
        batch, twice_channels, height, width = value.shape
        channels, spatial = twice_channels // 2, height * width
        size = batch * channels * spatial
        if size <= 0 or size >= 2**30:
            raise ValueError('unsupported empty or oversized tensor')
        output = torch.empty((batch, channels, height, width), device=value.device, dtype=value.dtype)
        args = [C.c_void_p(value.data_ptr()), C.c_void_p(output.data_ptr()), C.c_int(batch), C.c_int(channels), C.c_int(spatial)]
        params = (C.c_void_p * len(args))(*[C.cast(C.byref(arg), C.c_void_p).value for arg in args])
        stream = C.c_void_p(torch.cuda.current_stream(value.device).cuda_stream)
        self._check(self.launch(self.function, (size+255)//256, 1, 1, 256, 1, 1, 0, stream, params, None), 'cuLaunchKernel')
        return output
