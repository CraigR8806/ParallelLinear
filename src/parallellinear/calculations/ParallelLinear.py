from parallellinear.calculations.LinearCalculations import LinearCalculations
import pyopencl as cl
import numpy as np
import os


class ParallelLinear(LinearCalculations):


    def __init__(self, deferToNumpy = False, programs = None, ctx = None, queue = None, baseProgramNames=None):
        if programs is not None and ctx is not None and queue is not None and baseProgramNames is not None and type(ctx) == cl.Context and type(queue) == cl.CommandQueue:
            self.programsLoaded = True
            self.programs = programs
            self.ctx = ctx
            self.queue = queue
            self.baseProgramNames = baseProgramNames
        else:
            raise ValueError("ParalleLinear object not initialized properly.  Use ParallelLinear.loadPrograms(), or ParallelLinear.deferToNumpy() to initialize")
        self.deferToNumpy = deferToNumpy
        
    @classmethod
    def getLinearCalculator(cls, pyopencl_ctx=":2"):
        return cls._loadPrograms(cls, pyopencl_ctx)

    def _loadPrograms(cls, pyopencl_ctx=":2"):
        os.environ["PYOPENCL_CTX"] = pyopencl_ctx
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        prg = cl.Program(ctx, """
        __kernel void testing(const unsigned int size, __global float* a_g, __global const float* b_g){
            unsigned int gid1 = get_global_id(0);
            unsigned int gid2 = get_global_id(1);
            a_g[gid1 + size * gid2] = a_g[gid1 + size * gid2] + b_g[gid1 + size * gid2];
        }
        __kernel void add(__global const float *a_g, __global const float *b_g, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] + b_g[gid];
        }
        __kernel void sub(__global const float *a_g, __global const float *b_g, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] - b_g[gid];
        }
        __kernel void element_wise_multiply(__global const float *a_g, __global const float *b_g, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] * b_g[gid];
        }
        __kernel void add_scaler( __global float *a_g, const float scaler, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] + scaler;
        }
        __kernel void sub_scaler( __global float *a_g, const float scaler, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] - scaler;
        }
        __kernel void scaler_sub_from( __global float *a_g, const float scaler, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = scaler - a_g[gid];
        }
        __kernel void scale( __global const float *a_g, const float scaler, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] * scaler;
        }
        __kernel void descale(__global const float *a_g, const float scaler, __global float *res_g){
            int gid = get_global_id(0);
            res_g[gid] = a_g[gid] / scaler;
        }

        __kernel void add_inplace(__global float *a_g, __global const float *b_g){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] + b_g[gid];
        }
        __kernel void sub_inplace( __global float *a_g, __global const float *b_g){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] - b_g[gid];
        }
        __kernel void element_wise_multiply_inplace(__global float *a_g, __global const float *b_g){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] * b_g[gid];
        }
        __kernel void add_scaler_inplace( __global float *a_g, const float scaler){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] + scaler;
        }
        __kernel void sub_scaler_inplace( __global float *a_g, const float scaler){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] - scaler;
        }
        __kernel void scaler_sub_from_inplace( __global float *a_g, const float scaler){
            int gid = get_global_id(0);
            a_g[gid] = scaler - a_g[gid];
        }
        __kernel void scale_inplace( __global float *a_g, const float scaler){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] * scaler;
        }
        __kernel void descale_inplace(__global float *a_g, const float scaler){
            int gid = get_global_id(0);
            a_g[gid] = a_g[gid] / scaler;
        }

        __kernel void multiply(const unsigned int ac, const unsigned int bc, __global float * a, __global float * b, __global float * r) {
            int i = get_global_id(0); 
            r[i] = 0;
            for (unsigned int k = 0; k < ac; k++) {
                r[i] += a[k+((i/bc)*ac)] * b[(k*bc) + (i%bc)];
            }
        }
        """).build()

        programs = {}
        programs['testing'] = prg.testing
        programs['add'] = prg.add
        programs['subtract'] = prg.sub
        programs['element_wise_multiply'] = prg.element_wise_multiply
        programs['add_scaler'] = prg.add_scaler
        programs['sub_scaler'] = prg.sub_scaler
        programs['scaler_sub_from'] = prg.scaler_sub_from
        programs['scale'] = prg.scale
        programs['descale'] = prg.descale
        programs['add_inplace'] = prg.add_inplace
        programs['subtract_inplace'] = prg.sub_inplace
        programs['element_wise_multiply_inplace'] = prg.element_wise_multiply_inplace
        programs['add_scaler_inplace'] = prg.add_scaler_inplace
        programs['sub_scaler_inplace'] = prg.sub_scaler_inplace
        programs['scaler_sub_from_inplace'] = prg.scaler_sub_from_inplace
        programs['scale_inplace'] = prg.scale_inplace
        programs['descale_inplace'] = prg.descale_inplace
        programs['multiply'] = prg.multiply

        
        baseProgramNames = list(programs.keys())
        return cls(programs=programs, ctx=ctx, queue=queue, baseProgramNames=baseProgramNames)
    

    def loadCustomFunction(self, func_name, func):
        if func_name in self.programs:
            return
        
        func=func.replace("$i", "in[i]")

        prg = cl.Program(self.ctx, """
        __kernel void custom_func(
            __global float *in)
        {
            int i = get_global_id(0);
        """+func+"""    
        }
        """).build()

        self.programs[func_name] = prg.custom_func
        return func_name


    def unloadCustomFunction(self, func_name):
        if func_name in self.baseProgramNames:
            raise ValueError(func_name + " is already a standard program that cannot be unloaded")
        del self.programs[func_name]

    def testing(self, a, b):
        mf = cl.mem_flags

        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        self.programs['testing'](self.queue, a.shape, None, np.int32(len(a)), a_g, b_g)
        cl.enqueue_copy(self.queue, a, a_g)

    def _addInPlace(self, a, b):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        self.programs['add_inplace'](self.queue, a.shape, None, a_g, b_g)
        cl.enqueue_copy(self.queue, a, a_g)

    def _subInPlace(self, a, b):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        self.programs['subtract_inplace'](self.queue, a.shape, None, a_g, b_g)
        cl.enqueue_copy(self.queue, a, a_g)

    def _elementWiseMultiplyInPlace(self, a, b):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        self.programs['element_wise_multiply_inplace'](self.queue, a.shape, None, a_g, b_g)
        cl.enqueue_copy(self.queue, a, a_g)

    def _addScalerInPlace(self, a, scaler):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

        self.programs['add_scaler_inplace'](self.queue, a.shape, None, a_g, np.float32(scaler))
        cl.enqueue_copy(self.queue, a, a_g)


    def _subScalerInPlace(self, a, scaler):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

        self.programs['sub_scaler_inplace'](self.queue, a.shape, None, a_g, np.float32(scaler))
        cl.enqueue_copy(self.queue, a, a_g)

    def _subScalerFromInPlace(self, a, scaler):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

        self.programs['scaler_sub_from_inplace'](self.queue, a.shape, None, a_g, np.float32(scaler))
        cl.enqueue_copy(self.queue, a, a_g)

    def _scaleInPlace(self, a, scaler):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

        self.programs['scale_inplace'](self.queue, a.shape, None, a_g, np.float32(scaler))
        cl.enqueue_copy(self.queue, a, a_g)

    def _descaleInPlace(self, a, scaler):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

        self.programs['descale_inplace'](self.queue, a.shape, None, a_g, np.float32(scaler))
        cl.enqueue_copy(self.queue, a, a_g)

    def _add(self, a, b):


        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['add'](self.queue, a.shape, None, a_g, b_g, r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _sub(self, a, b):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['subtract'](self.queue, a.shape, None, a_g, b_g, r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _elementWiseMultiply(self, a, b):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['element_wise_multiply'](self.queue, a.shape, None, a_g, b_g, r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _addScaler(self, a, scaler):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['add_scaler'](self.queue, a.shape, None, a_g, np.float32(scaler), r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _subScaler(self, a, scaler):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['sub_scaler'](self.queue, a.shape, None, a_g, np.float32(scaler), r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _subScalerFrom(self, a, scaler):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['scaler_sub_from'](self.queue, a.shape, None, a_g, np.float32(scaler), r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np


    def _scale(self, a, scaler):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['scale'](self.queue, a.shape, None, a_g, np.float32(scaler), r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _descale(self, a, scaler):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs['descale'](self.queue, a.shape, None, a_g, np.float32(scaler), r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np

    def _multiply(self, a, b, a_rows:int, a_cols:int, b_rows:int, b_cols:int):

        res_np = np.empty(int(a_rows * b_cols), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        res_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, res_np.nbytes)

        self.programs['multiply'](self.queue, res_np.shape, None, np.int32(a_cols), np.int32(b_cols), a_g, b_g, res_g)

        cl.enqueue_copy(self.queue, res_np, res_g)
        return res_np

    def _sum(self, a):
        return a.sum()

    def _applyCustomFunctionInPlace(self, a, func_name):

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

        self.programs[func_name](self.queue, a.shape, None, a_g)

        cl.enqueue_copy(self.queue, a, a_g)

    def _applyCustomFunction(self, a, func_name):

        r_np=np.empty(len(a), dtype=a.dtype)

        mf = cl.mem_flags
        a_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        r_g = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

        self.programs[func_name](self.queue, a.shape, None, a_g, r_g)
        cl.enqueue_copy(self.queue, r_np, r_g)
        return r_np