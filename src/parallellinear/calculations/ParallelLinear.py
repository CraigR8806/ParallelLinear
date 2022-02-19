import pyopencl as cl
import numpy as np
import os

os.environ["PYOPENCL_CTX"] = ":2"

ctx = cl.create_some_context()
programs = {}
baseProgramNames = []

baseProgramsLoaded = False


def loadPrograms():
    if globals()['baseProgramsLoaded']:
        return
    currentPrograms=list(programs.keys())

    prg = cl.Program(ctx, """
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

    
    baseProgramNames = [x for x in list(programs.keys()) if x not in currentPrograms]
    
    globals()['baseProgramsLoaded'] = True
    

def loadCustomFunction(func_name, func):
    if func_name in programs:
        return
    
    func=func.replace("$i", "in[i]")

    prg = cl.Program(ctx, """
    __kernel void custom_func(
         __global float *in)
    {
        int i = get_global_id(0);
    """+func+"""    
    }
    """).build()

    programs[func_name] = prg.custom_func
    return func_name


def unloadCustomFunction(func_name):
    if func_name in baseProgramNames:
        raise ValueError(func_name + " is already a standard program that cannot be unloaded")
    del programs[func_name]


def _addInPlace(a, b):
    if a.getNumberOfRows() != b.getNumberOfRows() or a.getNumberOfColumns() != b.getNumberOfColumns():
        raise ValueError("When adding two Matricies, they need to be same dimensions")

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData().astype(np.float32))

    programs['add_inplace'](queue, a.getData().shape, None, a_g, b_g)
    cl.enqueue_copy(queue, a.getData(), a_g)

def _subInPlace(a, b):
    if a.getNumberOfRows() != b.getNumberOfRows() or a.getNumberOfColumns() != b.getNumberOfColumns():
        raise ValueError("When subtracting two Matricies, they need to be same dimensions")

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData().astype(np.float32))

    programs['subtract_inplace'](queue, a.getData().shape, None, a_g, b_g)
    cl.enqueue_copy(queue, a.getData(), a_g)

def _elementWiseMultiplyInPlace(a, b):
    if a.getNumberOfRows() != b.getNumberOfRows() or a.getNumberOfColumns() != b.getNumberOfColumns():
        raise ValueError("When subtracting two Matricies, they need to be same dimensions")

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData().astype(np.float32))

    programs['element_wise_multiply_inplace'](queue, a.getData().shape, None, a_g, b_g)
    cl.enqueue_copy(queue, a.getData(), a_g)

def _addScalerInPlace(a, scaler):

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))

    programs['add_scaler_inplace'](queue, a.getData().shape, None, a_g, np.float32(scaler))
    cl.enqueue_copy(queue, a.getData(), a_g)


def _subScalerInPlace(a, scaler):

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))

    programs['sub_scaler_inplace'](queue, a.getData().shape, None, a_g, np.float32(scaler))
    cl.enqueue_copy(queue, a.getData(), a_g)

def _subScalerFromInPlace(a, scaler):

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))

    programs['scaler_sub_from_inplace'](queue, a.getData().shape, None, a_g, np.float32(scaler))
    cl.enqueue_copy(queue, a.getData(), a_g)

def _scaleInPlace(a, scaler):

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))

    programs['scale_inplace'](queue, a.getData().shape, None, a_g, np.float32(scaler))
    cl.enqueue_copy(queue, a.getData(), a_g)

def _descaleInPlace(a, scaler):

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))

    programs['descale_inplace'](queue, a.getData().shape, None, a_g, np.float32(scaler))
    cl.enqueue_copy(queue, a.getData(), a_g)

def _add(a, b):
    if a.getNumberOfRows() != b.getNumberOfRows() or a.getNumberOfColumns() != b.getNumberOfColumns():
        raise ValueError("When adding two Matricies, they need to be same dimensions")

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['add'](queue, a.getData().shape, None, a_g, b_g, r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _sub(a, b):
    if a.getNumberOfRows() != b.getNumberOfRows() or a.getNumberOfColumns() != b.getNumberOfColumns():
        raise ValueError("When adding two Matricies, they need to be same dimensions")

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['subtract'](queue, a.getData().shape, None, a_g, b_g, r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _elementWiseMultiply(a, b):
    if a.getNumberOfRows() != b.getNumberOfRows() or a.getNumberOfColumns() != b.getNumberOfColumns():
        raise ValueError("When adding two Matricies, they need to be same dimensions")

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['element_wise_multiply'](queue, a.getData().shape, None, a_g, b_g, r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _addScaler(a, scaler):

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['add_scaler'](queue, a.getData().shape, None, a_g, np.float32(scaler), r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _subScaler(a, scaler):

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['sub_scaler'](queue, a.getData().shape, None, a_g, np.float32(scaler), r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _subScalerFrom(a, scaler):

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['scaler_sub_from'](queue, a.getData().shape, None, a_g, np.float32(scaler), r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np


def _scale(a, scaler):

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['scale'](queue, a.getData().shape, None, a_g, np.float32(scaler), r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _descale(a, scaler):

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(int(a.getNumberOfRows() * a.getNumberOfColumns())).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData().astype(np.float32))
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs['descale'](queue, a.getData().shape, None, a_g, np.float32(scaler), r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np

def _multiply(a, b):
    if a.getNumberOfColumns() != b.getNumberOfRows():
        raise ValueError("When muliplying two matricies, the m*n n*p rule needs to be followed...")

    queue = cl.CommandQueue(ctx)

    res_np = np.empty(int(a.getNumberOfRows() * b.getNumberOfColumns()), dtype=np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData())
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b.getData())
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

    programs['multiply'](queue, res_np.shape, None, np.int32(a.getNumberOfColumns()), np.int32(b.getNumberOfColumns()), a_g, b_g, res_g)

    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

def _applyCustomFunctionInPlace(a, func_name):

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a.getData())

    programs[func_name](queue, a.getData().shape, None, a_g)

    cl.enqueue_copy(queue, a.getData(), a_g)

def _applyCustomFunction(a, func_name):

    queue = cl.CommandQueue(ctx)

    r_np=np.empty(a.getNumberOfRows() * a.getNumberOfColumns()).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a.getData())
    r_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)

    programs[func_name](queue, a.getData().shape, None, a_g, r_g)
    cl.enqueue_copy(queue, r_np, r_g)
    return r_np