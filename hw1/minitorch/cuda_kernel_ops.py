from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    shape_broadcast,
)
from .tensor_ops import MapProto, TensorOps
import os
import ctypes
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit

# Load the shared library
try:
    lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
except:
    print("cuda kernels not implemented: combine.so not found")

datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Define the argument type for the tensorMap function
            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            # Define the return type for the tensorMap function
            lib.tensorMap.restype = None

            # Call the function
            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            # Define the argument type for the tensorZip function
            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            # Define the return type for the tensorZip function
            lib.tensorZip.restype = None

            # BEGIN ASSIGN1_2
            # TODO
            lib.tensorZip(
                out._tensor._storage,                       # out_storage
                out._tensor._shape.astype(np.int32),        # out_shape
                out._tensor._strides.astype(np.int32),      # out_strides
                out.size,                                   # out_size
                len(out.shape),                             # out_shape_size
                a._tensor._storage,                         # a_storage
                a._tensor._shape.astype(np.int32),          # a_shape
                a._tensor._strides.astype(np.int32),        # a_strides
                a.size,                                     # a_size
                len(a.shape),                               # a_shape_size
                b._tensor._storage,                         # b_storage     
                b._tensor._shape.astype(np.int32),          # b_shape
                b._tensor._strides.astype(np.int32),        # b_strides
                b.size,                                     # b_size
                len(b.shape),                               # b_shape_size
                fn_id,                                      # fn_id
            )
            # raise NotImplementedError("Zip Function Not Implemented Yet")
            # END ASSIGN1_2
            
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], reduce_value: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            # Define the return type for the tensorReduce function
            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            # Define the return type for the tensorReduce function
            lib.tensorReduce.restype = None

            # BEGIN ASSIGN1_2
            # TODO
            # 1. Call the tensorReduce function implemented in CUDA

            lib.tensorReduce(
                out._tensor._storage,                       # out_storage
                out._tensor._shape.astype(np.int32),        # out_shape
                out._tensor._strides.astype(np.int32),      # out_strides
                out.size,                                   # out_size
                a._tensor._storage,                         # in_storage
                a._tensor._shape.astype(np.int32),          # in_shape
                a._tensor._strides.astype(np.int32),        # in_strides
                dim,                                        # reduce_dim
                reduce_value,                              # reduce_value
                len(a.shape),                               # shape_len
                fn_id,                                      # fn_id
            )
            
            # raise NotImplementedError("Reduce Function Not Implemented Yet")
            # END ASSIGN1_2
            
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Track if both inputs were originally 2D to reshape output later
        both_2d = 0
        if len(a.shape) == 2:  # a: [R_a, C_a] -> [1, R_a, C_a]
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:  # b: [R_b, C_b] -> [1, R_b, C_b]
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        # Broadcast batch dimensions and compute output shape
        # a: [...batch_dims, R_a, C_a], b: [...batch_dims, R_b, C_b]
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))  # Broadcast batch dims
        ls.append(a.shape[-2])  # R_a (rows from a)
        ls.append(b.shape[-1])  # C_b (cols from b)
        assert a.shape[-1] == b.shape[-2]  # C_a == R_b (inner dimensions must match)
        out = a.zeros(tuple(ls))  # out: [...batch_dims, R_a, C_b]

        # Handle high-dimensional tensors by flattening batch dims
        # Example: [B1, B2, R, C] -> [B1*B2, R, C]
        more_3d = False
        if len(out.shape) > 3:  # out: [B1, B2, ..., R_a, C_b] -> [B_flat, R_a, C_b]
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
        if len(a.shape) > 3:  # a: [B1, B2, ..., R_a, C_a] -> [B_flat, R_a, C_a]
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:  # b: [B1, B2, ..., R_b, C_b] -> [B_flat, R_b, C_b]
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        # All tensors now have shape [B, *, *] with same batch size B
        assert a.shape[0] == b.shape[0]    # B_a == B_b
        assert a.shape[0] == out.shape[0]  # B_a == B_out

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        # Define the return type for the tensorZip function
        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        # BEGIN ASSIGN1_2
        # TODO
        # 1. Call the Matmul function implemented in CUDA
        lib.MatrixMultiply(
            out._tensor._storage,                       # out_storage
            out._tensor._shape.astype(np.int32),        # out_shape
            out._tensor._strides.astype(np.int32),      # out_strides
            a._tensor._storage,                         # a_storage
            a._tensor._shape.astype(np.int32),          # a_shape
            a._tensor._strides.astype(np.int32),        # a_strides
            b._tensor._storage,                         # b_storage
            b._tensor._shape.astype(np.int32),          # b_shape
            b._tensor._strides.astype(np.int32),        # b_strides
            a.shape[0],                                 # batch_size
            out.shape[1],                               # out_shape[1], m
            out.shape[2]                                # out_shape[2], p
        )

        # raise NotImplementedError("Matrix Multiply Function Not Implemented Yet")
        # END ASSIGN1_2
        
        # Restore original dimensionality
        if both_2d:  # Remove batch dim if both inputs were 2D: [1, R_a, C_b] -> [R_a, C_b]
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:  # Unflatten batch dims: [B_flat, R_a, C_b] -> [B1, B2, ..., R_a, C_b]
            out = out.view(*ls)
        return out
