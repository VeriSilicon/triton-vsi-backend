import functools
import hashlib
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Optional, Any, Tuple, NamedTuple, List, Dict, Callable

import torch

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.compiler.compiler import ASTSource
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager

from .compiler import get_dump_dir

_BACKEND_ROOT_DIR = Path(__file__).parent.resolve()

_TRITON_TO_C_TYPE_MAP = {
    "i1": "int32_t",  # TODO(i1): Revisit i1.
    "u1": "uint32_t",
    "i8": "int8_t",
    "u8": "uint8_t",
    "i16": "int16_t",
    "u16": "uint16_t",
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
    "fp16": "__fp16",
    "bf16": "__bf16",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
}

_PY_ARG_TYPE_TO_FORMAT_MAP = {
    "PyObject*": "O",
    "int8_t": "b",
    "uint8_t": "B",
    "int16_t": "h",
    "uint16_t": "H",
    "int32_t": "i",
    "uint32_t": "I",
    "int64_t": "L",
    "uint64_t": "K",
    "long": "l",
    "size_t": "k",
    "float": "f",
    "double": "d",
}


@functools.lru_cache()
def _get_sdk_dir() -> Path:
    sdk_dir = os.getenv("VSI_SDK_DIR", "")
    return Path(sdk_dir)


@functools.lru_cache()
def _get_library_dirs() -> List[Path]:
    lib_dirs = [_BACKEND_ROOT_DIR / "lib"]

    sdk_dir = _get_sdk_dir()
    if sdk_dir:
        lib_dirs.append(sdk_dir / "lib")
        lib_dirs.append(sdk_dir / "drivers")

    return lib_dirs


@functools.lru_cache()
def _get_include_dirs() -> List[Path]:
    inc_dirs = [_BACKEND_ROOT_DIR / "include"]

    sdk_dir = _get_sdk_dir()
    if sdk_dir:
        inc_dirs.append(sdk_dir / "include")

    return inc_dirs


def _load_module_from_so(so_path: Path, name: str) -> ModuleType:
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compile_module_from_src(src: str,
                             name: str,
                             dump_dir: Optional[Path] = None) -> ModuleType:
    library_dirs = _get_library_dirs()
    include_dirs = _get_include_dirs()
    libraries = ["OpenCL"]

    if dump_dir:
        src_path = dump_dir / f"{name}.c"
        src_path.write_text(src)
        so_path = _build(name, src_path, src_path.parent, library_dirs,
                         include_dirs, libraries)
        return _load_module_from_so(so_path, name)
    else:
        key = hashlib.sha256(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        cached_so_path = cache.get_file(f"{name}.so")
        if cached_so_path is None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                src_path = tmp_dir / f"{name}.c"
                src_path.write_text(src)
                so_path = _build(name, src_path, src_path.parent, library_dirs,
                                 include_dirs, libraries)
                with open(so_path, "rb") as f:
                    cached_so_path = cache.put(f.read(),
                                               f"{name}.so",
                                               binary=True)
        return _load_module_from_so(cached_so_path, name)


class VSIUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(VSIUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = (_BACKEND_ROOT_DIR / "driver.c").read_text()
        self.mod = _compile_module_from_src(src, "vsi_utils")

    def load_binary(self, name: str, binary_data: bytes, shared: int,
                    device_index: int) -> Tuple[int, int, int, int]:
        import torch.vsi
        raw_context = torch.vsi._get_raw_context()
        raw_device = torch.vsi._get_raw_device(device_index)
        return self.mod.load_binary(raw_context, raw_device, name, binary_data,
                                    shared)

    def get_device_properties(self, device_index: int) -> Dict[str, Any]:
        import torch.vsi
        prop = torch.vsi.get_device_properties(device_index)
        return {
            "multiprocessor_count": prop.multi_processor_count,
            "warpSize": prop.warp_size,
            "max_shared_mem": prop.local_memory,
        }


def _triton_to_cl_type(triton_type: str) -> str:
    if triton_type[0] == '*':
        return "cl_mem"
    return _TRITON_TO_C_TYPE_MAP[triton_type]


def _triton_to_c_scalar_type(triton_type: str) -> str:
    if triton_type[0] == '*':
        pointee_type = _TRITON_TO_C_TYPE_MAP[triton_type[1:]]
        return pointee_type
    return _TRITON_TO_C_TYPE_MAP[triton_type]


def _triton_to_c_type(triton_type: str) -> str:
    c_scalar_type = _triton_to_c_scalar_type(triton_type)
    if triton_type[0] == '*':
        return f"{c_scalar_type}*"
    return c_scalar_type


def _triton_to_py_arg_type(triton_type: str) -> str:
    if triton_type == "constexpr":
        return "PyObject*"
    if triton_type[0] == '*':
        return "PyObject*"
    return _TRITON_TO_C_TYPE_MAP[triton_type]


def _get_py_arg_format(c_type: str) -> str:
    return _PY_ARG_TYPE_TO_FORMAT_MAP[c_type]


def _make_launcher(signature: Dict[int, str],
                   constants: Dict[Tuple[int, ...], int | float | bool],
                   metadata: NamedTuple) -> str:
    constants_indices = [i for i, t in signature.items() if t == "constexpr"]
    params = [i for i in signature.keys() if i not in constants_indices]
    tensor_params = [i for i in params if signature[i][0] == '*']
    scalar_params = [i for i in params if signature[i][0] != '*']

    NEW_LINE = '\n'
    # Generate glue code.
    src = f"""
#include <stdbool.h>
#include <dlfcn.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <Python.h>

static inline void oclAssert(cl_int code, const char *file, int line) {{
  if (code == CL_SUCCESS) {{
    return;
  }}

  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();

  PyErr_Format(PyExc_RuntimeError,
               "Triton Error [OpenCL]: code: %d, at %s:%d", code, file, line);

  PyGILState_Release(gil_state);
}}

#define OCL_CHECK(ans) {{ oclAssert((ans), __FILE__, __LINE__); }}

typedef struct {{
  cl_mem dev_ptr;
  bool valid;
}} device_ptr_info_t;

// TODO(TC): Do not query tensor shape at runtime.
#define TC_MAX_TENSOR_RANK 4
typedef struct {{
  int32_t rank;
  size_t shape[TC_MAX_TENSOR_RANK];
}} tensor_shape_info_t;

static device_ptr_info_t getPointer(PyObject *obj, int index) {{
  device_ptr_info_t ptr_info = {{
    .dev_ptr = NULL,
    .valid = true
  }};

  if (obj == Py_None) {{
    // valid nullptr.
    return ptr_info;
  }}

  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (cl_mem)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}

  PyObject *get_ptr_method = PyObject_GetAttrString(obj, "data_ptr");
  if(!get_ptr_method){{
    PyErr_SetString(PyExc_AttributeError, "Object has no data_ptr attribute");
    return ptr_info;
  }}

  PyObject *empty_tuple = PyTuple_New(0);
  PyObject *ret = PyObject_Call(get_ptr_method, empty_tuple, NULL);
  Py_DECREF(empty_tuple);
  Py_DECREF(get_ptr_method);
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    return ptr_info;
  }}

  ptr_info.dev_ptr = (cl_mem)PyLong_AsVoidPtr(ret);
  if(!ptr_info.dev_ptr) {{
    PyErr_Format(PyExc_ValueError, "Pointer arg#%d cannot be accessed from Triton", index);
    ptr_info.valid = false;
  }}

  Py_DECREF(ret);
  return ptr_info;
}}

static tensor_shape_info_t getTensorShape(PyObject* obj) {{
  tensor_shape_info_t info = {{
    .rank = -1,
    .shape = {{1, 1, 1, 1}}
  }};

  PyObject *shape_tuple = PyObject_GetAttrString(obj, "shape");
  if (!shape_tuple) {{
    PyErr_SetString(PyExc_AttributeError, "Tensor object has no shape attribute");
    return info;
  }}

  if (!PyTuple_Check(shape_tuple)) {{
    PyErr_SetString(PyExc_TypeError, "Tensor shape is not a tuple");

    Py_DECREF(shape_tuple);
    return info;
  }}

  int32_t rank = PyTuple_Size(shape_tuple);
  if (rank > TC_MAX_TENSOR_RANK) {{
    PyErr_Format(PyExc_TypeError,
                 "Tensor with rank %d > %d is not supported by TC", rank, TC_MAX_TENSOR_RANK);
    Py_DECREF(shape_tuple);
    return info;
  }}

  info.rank = rank;
  int32_t pad_offset = TC_MAX_TENSOR_RANK - rank;
  for (int32_t i = 0; i < rank; i++) {{
    PyObject* dim = PyTuple_GetItem(shape_tuple, i);
    if (!PyLong_Check(dim)) {{
      PyErr_SetString(PyExc_TypeError, "Tensor shape dimensions must be integers");
      Py_DECREF(shape_tuple);
      return info;
    }}
    info.shape[pad_offset + i] = PyLong_AsLong(dim);
  }}

  Py_DECREF(shape_tuple);
  return info;
}}

static void _launch(int grid_dim_x, int grid_dim_y, int grid_dim_z, int num_warps,
                    cl_command_queue queue,
                    cl_kernel kernel,
                    {", ".join(f"{_triton_to_cl_type(signature[i])} arg_{i}" for i in params)},
                    cl_mem* d_mm_commands, size_t mm_num,
                    cl_mem* d_dma_commands, size_t dma_num) {{
  uint32_t arg_index = 0;
  // Set kernel args.
{
  NEW_LINE.join(
    f"  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof({_triton_to_cl_type(signature[i])}), &arg_{i}));"
    for i in params
  )
}

  // Set extra kernel args.
  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(int), &grid_dim_x));
  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(int), &grid_dim_y));
  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(int), &grid_dim_z));

  // Set TC command args.
  for (size_t i = 0; i < mm_num; ++i) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &d_mm_commands[i]));
  }}
  for (size_t i = 0; i < dma_num; ++i) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &d_dma_commands[i]));
  }}

  if (grid_dim_x * grid_dim_y * grid_dim_z > 0) {{
    size_t local_work_size[] = {{
      {f"(size_t)num_warps * {metadata.target.warp_size}" if metadata.num_warps > 0 else "1"},
      1,
      1,
    }};
    size_t global_work_size[] = {{
      (size_t)grid_dim_x * local_work_size[0],
      (size_t)grid_dim_y * local_work_size[1],
      (size_t)grid_dim_z * local_work_size[2],
    }};
    OCL_CHECK(clEnqueueNDRangeKernel(
      queue,
      kernel,
      /*work_dim=*/3,
      /*global_work_offset=*/NULL,
      global_work_size,
      local_work_size,
      /*num_events_in_wait_list=*/0,
      /*event_wait_list=*/NULL,
      /*event=*/NULL
    ));
  }} else {{
    PyErr_Format(PyExc_RuntimeError,
                 "Invalid grid size: (%d, %d, %d)\\n", grid_dim_x, grid_dim_y, grid_dim_z);
  }}
}}

// Declaration pf kernel_cmd_gen.so APIs:
typedef size_t (*get_tc_num_func_t)(void);
typedef bool (*generate_matmul_command_func_t)(
  size_t index,
  const size_t* grid_size,
  {", ".join(f"{_triton_to_c_type(signature[i])} arg_{i}" for i in params)},
  int grid_dim_x, int grid_dim_y, int grid_dim_z,
  {", ".join(f"const size_t* arg_{i}_shape" for i in tensor_params)},
  int8_t* cmds_out
);
typedef bool (*generate_dma_command_func_t)(
  size_t index,
  {", ".join(f"{_triton_to_c_type(signature[i])} arg_{i}" for i in params)},
  int grid_dim_x, int grid_dim_y, int grid_dim_z,
  {", ".join(f"const size_t* arg_{i}_shape" for i in tensor_params)},
  int8_t* cmds_out
);

static PyObject* launch(PyObject* self, PyObject* args) {{
  int grid_dim_x, grid_dim_y, grid_dim_z;
  uint64_t raw_stream;
  uint64_t raw_kernel;
  PyObject* launch_enter_hook = NULL;
  PyObject* launch_exit_hook = NULL;
  PyObject* kernel_metadata = NULL;
  PyObject* launch_metadata = NULL;

{
  NEW_LINE.join([f"  {_triton_to_py_arg_type(t)} _arg_{i};"
                 for i, t in signature.items()])
}

  if(!PyArg_ParseTuple(args,
    "{"iiiKKOOOO" + ''.join(
        _get_py_arg_format(_triton_to_py_arg_type(t))
        for t in signature.values())
    }",
    &grid_dim_x, &grid_dim_y, &grid_dim_z,
    &raw_stream,
    &raw_kernel,
    &kernel_metadata,
    &launch_metadata,
    &launch_enter_hook,
    &launch_exit_hook,
    {", ".join(f"&_arg_{i}" for i in signature.keys())}
  )) {{
    return NULL;
  }}

  // Extract kernel metadata.
  int num_warps, num_ctas, shared_memory, cluster_dim_x, cluster_dim_y, cluster_dim_z;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &cluster_dim_x, &cluster_dim_y, &cluster_dim_z)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  // Extract launch metadata.
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  cl_int err = CL_SUCCESS;

  cl_context context;
  cl_command_queue queue = (cl_command_queue)(raw_stream);
  cl_kernel kernel = (cl_kernel)(raw_kernel);

  // Get CL context.
  OCL_CHECK(clGetCommandQueueInfo(
    queue,
    CL_QUEUE_CONTEXT,
    sizeof(cl_context),
    &context,
    NULL
  ))

{
  NEW_LINE.join(f"  device_ptr_info_t ptr_info_{i} = getPointer(_arg_{i}, {i}); if (!ptr_info_{i}.valid) return NULL;"
                for i in tensor_params)
}

{
  NEW_LINE.join(f"  cl_mem d_arg_{i} = ptr_info_{i}.dev_ptr;"
                for i in tensor_params)
}

  // Generate TC commands.
  #define CMD_SIZE 256
  size_t num_blocks = grid_dim_x * grid_dim_y * grid_dim_z;
  size_t cmd_size_all_blocks = num_blocks * CMD_SIZE;

  void* handle = dlopen("{metadata.cmd_gen_path}", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {{
    return PyErr_Format(PyExc_RuntimeError,
                        "Failed to load {metadata.cmd_gen_path}, error: %s", dlerror());
  }}

  get_tc_num_func_t get_matmul_num_func =
    (get_tc_num_func_t)dlsym(handle, "get_matmul_num");
  if (!get_matmul_num_func) {{
    dlclose(handle);
    return PyErr_Format(PyExc_RuntimeError,
                        "Failed to resolve symbol: `get_matmul_num`, error: %s", dlerror());
  }}

  get_tc_num_func_t get_dma_num_func =
    (get_tc_num_func_t)dlsym(handle, "get_dma_num");
  if (!get_dma_num_func) {{
    dlclose(handle);
    return PyErr_Format(PyExc_RuntimeError,
                        "Failed to resolve symbol: `get_dma_num`, error: %s", dlerror());
  }}

  {
    NEW_LINE.join(f"    tensor_shape_info_t shape_info_{i} = getTensorShape(_arg_{i}); if (shape_info_{i}.rank == -1) return NULL;"
                  for i in tensor_params)
  }

  size_t mm_num = get_matmul_num_func();
  int8_t** h_mm_commands = NULL;
  cl_mem* d_mm_commands = NULL;

  if (mm_num > 0) {{
    generate_matmul_command_func_t generate_matmul_command_func =
      (generate_matmul_command_func_t)dlsym(handle, "generate_matmul_command");
    if (!generate_matmul_command_func) {{
      dlclose(handle);
      return PyErr_Format(PyExc_RuntimeError,
                          "Failed to resolve symbol: `generate_matmul_command`, error: %s", dlerror());
    }}

    h_mm_commands = (int8_t**)malloc(mm_num * sizeof(int8_t*));
    d_mm_commands = (cl_mem*)malloc(mm_num * sizeof(cl_mem));

    // TODO(TC): Pass grid_size by values.
    size_t grid_size[3] = {{
      (size_t)grid_dim_x,
      (size_t)grid_dim_y,
      (size_t)grid_dim_z,
    }};

    for (size_t i = 0; i < mm_num; ++i) {{
      h_mm_commands[i] = (int8_t*)malloc(cmd_size_all_blocks);
      if (!generate_matmul_command_func(
        i, grid_size,
        {", ".join("NULL" if signature[i][0] == '*' else f"_arg_{i}" for i in params)},
        grid_dim_x, grid_dim_y, grid_dim_z,
        {", ".join(f"shape_info_{i}.shape" for i in tensor_params)},
        h_mm_commands[i])) {{
        return PyErr_Format(PyExc_RuntimeError, "Failed to generate the #%d TC mm command", i);
      }}

      d_mm_commands[i] = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cmd_size_all_blocks,
        h_mm_commands[i],
        &err
      );
      OCL_CHECK(err);
    }}
  }}

  size_t dma_num = get_dma_num_func();
  int8_t** h_dma_commands = NULL;
  cl_mem* d_dma_commands = NULL;

  if (dma_num > 0) {{
    generate_dma_command_func_t generate_dma_command_func =
      (generate_dma_command_func_t)dlsym(handle, "generate_dma_command");
    if (!generate_dma_command_func) {{
      dlclose(handle);
      return PyErr_Format(PyExc_RuntimeError,
                          "Failed to resolve symbol: `generate_dma_command`, error: %s", dlerror());
    }}

    h_dma_commands = (int8_t**)malloc(dma_num * sizeof(int8_t*));
    d_dma_commands = (cl_mem*)malloc(dma_num * sizeof(cl_mem));

    for (size_t i = 0; i < dma_num; ++i) {{
      h_dma_commands[i] = (int8_t*)malloc(cmd_size_all_blocks);
      if (!generate_dma_command_func(
        i,
        {", ".join("NULL" if signature[i][0] == '*' else f"_arg_{i}" for i in params)},
        grid_dim_x, grid_dim_y, grid_dim_z,
        {", ".join(f"shape_info_{i}.shape" for i in tensor_params)},
        h_dma_commands[i])) {{
        return PyErr_Format(PyExc_RuntimeError, "Failed to generate the #%d TC dma command", i);
      }}

      d_dma_commands[i] = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cmd_size_all_blocks,
        h_dma_commands[i],
        &err
      );
      OCL_CHECK(err);
    }}
  }}

  _launch(
    grid_dim_x, grid_dim_y, grid_dim_z, num_warps,
    queue, kernel,
    {", ".join(f"{'d' if signature[i][0] == '*' else ''}_arg_{i}" for i in params)},
    d_mm_commands, mm_num, d_dma_commands, dma_num
  );

  // TODO(TC): Manage cmd buffers in py side.
  if (d_mm_commands || d_dma_commands) {{
    OCL_CHECK(clFinish(queue));
  }}

  if (mm_num > 0) {{
    for (size_t i = 0; i < mm_num; ++i) {{
        free(h_mm_commands[i]);
        OCL_CHECK(clReleaseMemObject(d_mm_commands[i]));
    }}
    free(h_mm_commands);
    free(d_mm_commands);
  }}

  if (dma_num > 0) {{
    for (size_t i = 0; i < dma_num; ++i) {{
        free(h_dma_commands[i]);
        OCL_CHECK(clReleaseMemObject(d_dma_commands[i]));
    }}
    free(h_dma_commands);
    free(d_dma_commands);
  }}

  if(launch_exit_hook != Py_None) {{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};


PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


def _make_standalone_launcher(signature: Dict[int, str],
                              constants: Dict[Tuple[int, ...],
                                              int | float | bool],
                              metadata: NamedTuple) -> str:
    constants_indices = [i for i, t in signature.items() if t == "constexpr"]
    params = [i for i in signature.keys() if i not in constants_indices]
    tensor_params = [i for i in params if signature[i][0] == '*']
    scalar_params = [i for i in params if signature[i][0] != '*']

    NEW_LINE = '\n'
    # Generate standalone launcher code.
    src = f"""
#include <dlfcn.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "npy.hpp"

// Declaration pf kernel_cmd_gen.so APIs:
typedef size_t (*get_tc_num_func_t)(void);
typedef bool (*generate_matmul_command_func_t)(
  size_t index,
  const size_t* grid_size,
  {", ".join(f"{_triton_to_c_type(signature[i])} arg_{i}" for i in params)},
  int grid_dim_x, int grid_dim_y, int grid_dim_z,
  {", ".join(f"const size_t* arg_{i}_shape" for i in tensor_params)},
  int8_t* cmds_out
);
typedef bool (*generate_dma_command_func_t)(
  size_t index,
  {", ".join(f"{_triton_to_c_type(signature[i])} arg_{i}" for i in params)},
  int grid_dim_x, int grid_dim_y, int grid_dim_z,
  {", ".join(f"const size_t* arg_{i}_shape" for i in tensor_params)},
  int8_t* cmds_out
);

static inline void oclAssert(cl_int code, const char* file, int line) {{
  if (code == CL_SUCCESS) {{
    return;
  }}
  fprintf(stderr, "[ERROR][OpenCL] Code: %d, at %s:%d\\n", code, file, line);
  exit(code);
}}

#define OCL_CHECK(ans) {{ oclAssert((ans), __FILE__, __LINE__); }}

static void _launch(int grid_dim_x, int grid_dim_y, int grid_dim_z, int num_warps,
                    cl_command_queue queue, cl_kernel kernel,
                    {", ".join(f"{_triton_to_cl_type(signature[i])} arg_{i}" for i in params)},
                    cl_mem* d_mm_commands, size_t mm_num,
                    cl_mem* d_dma_commands, size_t dma_num) {{
  uint32_t arg_index = 0;
  // Set kernel args.
{
  NEW_LINE.join(
    f"  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof({_triton_to_cl_type(signature[i])}), &arg_{i}));"
    for i in params
  )
}
  // Set extra kernel args.
  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(int), &grid_dim_x));
  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(int), &grid_dim_y));
  OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(int), &grid_dim_z));

  // Set TC command args.
  for (size_t i = 0; i < mm_num; ++i) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &d_mm_commands[i]));
  }}
  for (size_t i = 0; i < dma_num; ++i) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), &d_dma_commands[i]));
  }}

  if (grid_dim_x * grid_dim_y * grid_dim_z > 0) {{
    size_t local_work_size[] = {{
        {f"(size_t)num_warps * {metadata.target.warp_size}" if metadata.num_warps > 0 else "1"},
        1,
        1,
    }};
    size_t global_work_size[] = {{
        (size_t)grid_dim_x * local_work_size[0],
        (size_t)grid_dim_y * local_work_size[1],
        (size_t)grid_dim_z * local_work_size[2],
    }};
    OCL_CHECK(clEnqueueNDRangeKernel(
      queue,
      kernel,
      /*work_dim=*/3,
      /*global_work_offset=*/NULL,
      global_work_size,
      local_work_size,
      /*num_events_in_wait_list=*/0,
      /*event_wait_list=*/NULL,
      /*event=*/NULL
    ));
  }} else {{
    fprintf(stderr, "Invalid grid size: (%d, %d, %d)",
            grid_dim_x, grid_dim_y, grid_dim_z);
  }}
}}

// clang-format off
static const struct option CLI_OPTIONS_FLAGS[] = {{
  {{"device",      required_argument,  NULL, 'd' }},
  {{"grids",       required_argument,  NULL, 'g' }},
  {{"num_warps",   required_argument,  NULL, 'w' }},
  {{"spv",         required_argument,  NULL, 'm' }},
  {{NULL,          0,                  NULL,  0  }},
}};
// clang-format on

typedef struct {{
  int device_index;
  int grids[3];
  int num_warps;
  const char* spv_path;
}} cli_option_t;

cli_option_t parseCLIOptions(int argc, char* argv[]) {{
  cli_option_t option = {{
      .device_index = 0,
      .grids = {{1, 1, 1}},
      .num_warps = {metadata.num_warps},
      .spv_path = "{metadata.name}.spv",
  }};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "d:g:w:m", CLI_OPTIONS_FLAGS,
                          &option_index)) != -1) {{
    switch (c) {{
      case 'd':
        sscanf(optarg, "%d", &option.device_index);
        break;
      case 'g':
        sscanf(optarg, "%d,%d,%d", &option.grids[0], &option.grids[1],
               &option.grids[2]);
        break;
      case 'w':
        sscanf(optarg, "%d", &option.num_warps);
        break;
      case 'm':
        option.spv_path = optarg;
        break;
      default:
        break;
    }}
  }}
  return option;
}}

int main(int argc, char* argv[]) {{
  cli_option_t option = parseCLIOptions(argc, argv);

  // Initialize OpenCL.
  cl_platform_id platform;
  OCL_CHECK(clGetPlatformIDs(1, &platform, nullptr));

  uint32_t num_devices;
  OCL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr,
                           &num_devices));

  if (num_devices == 0) {{
    fprintf(stderr, "[ERROR][OpenCL] No device found!\\n");
    exit(1);
  }}

  cl_device_id* devices =
      (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  OCL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices,
                           nullptr));

  cl_int err;
  cl_context context =
      clCreateContext(nullptr, num_devices, devices, nullptr, nullptr, &err);
  OCL_CHECK(err);

  if (option.device_index >= num_devices) {{
    fprintf(stderr, "[ERROR][OpenCL] Invalid device index: %d\\n",
            option.device_index);
    exit(1);
  }}
  cl_device_id device = devices[option.device_index];

  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, nullptr, &err);
  OCL_CHECK(err);

  // Load kernel from SPIRV.
  FILE* spv_file = fopen(option.spv_path, "rb");
  if (!spv_file) {{
    fprintf(stderr, "[ERROR] Failed to open SPIR-V file at %s\\n",
            option.spv_path);
    exit(ferror(spv_file));
  }}

  fseek(spv_file, 0, SEEK_END);
  size_t spv_size = ftell(spv_file);
  fseek(spv_file, 0, SEEK_SET);

  uint8_t* spv_data = (uint8_t*)malloc(spv_size);
  if (fread(spv_data, 1, spv_size, spv_file) != spv_size) {{
    fprintf(stderr, "[ERROR] Failed to read SPIR-V file\\n");
    exit(ferror(spv_file));
  }}

  cl_program program = clCreateProgramWithIL(context, spv_data, spv_size, &err);
  OCL_CHECK(err);

  OCL_CHECK(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));
  cl_kernel kernel = clCreateKernel(program, "{metadata.name}", &err);
  OCL_CHECK(err);

  fclose(spv_file);
  free(spv_data);

  // Load input args from npy files.
{
  NEW_LINE.join(
    f'  auto arg_{i}_npy = npy::read_npy<{_triton_to_c_scalar_type(signature[i])}>("inputs/arg_{i}.npy");'
    for i in params
  )
}

  // Extract tensor args.
{
  NEW_LINE.join(
    f"  {_triton_to_c_type(signature[i])} arg_{i} = arg_{i}_npy.data.data();"
    f"  size_t arg_{i}_size = arg_{i}_npy.data.size() * sizeof({_triton_to_c_scalar_type(signature[i])});"
    for i in tensor_params
  )
}
  // Extract scalar args.
{
  NEW_LINE.join(
    f"  {_triton_to_c_type(signature[i])} arg_{i} = arg_{i}_npy.data[0];"
    for i in scalar_params
  )
}

  // Create OpenCL buffer for tensor args.
{
  NEW_LINE.join(
    f"  cl_mem d_arg_{i} = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, arg_{i}_size, arg_{i}, &err);"
    " OCL_CHECK(err);"
    for i in tensor_params
  )
}

  int grid_dim_x = option.grids[0];
  int grid_dim_y = option.grids[1];
  int grid_dim_z = option.grids[2];
  int num_warps = option.num_warps;

  // Generate TC commands.
  #define CMD_SIZE 256
  size_t num_blocks = grid_dim_x * grid_dim_y * grid_dim_z;
  size_t cmd_size_all_blocks = num_blocks * CMD_SIZE;

  void* handle =
      dlopen("./{metadata.name}_cmd_gen.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {{
    fprintf(stderr,
            "[ERROR][CMD_GEN] Failed to load {metadata.name}_cmd_gen.so, "
            "error: %s\\n", dlerror());
    exit(errno);
  }}

  get_tc_num_func_t get_matmul_num_func =
      (get_tc_num_func_t)dlsym(handle, "get_matmul_num");
  if (!get_matmul_num_func) {{
    dlclose(handle);
    fprintf(stderr,
            "[ERROR][CMD_GEN] Failed to resolve symbol: `get_matmul_num`, "
            "error: %s", dlerror());
    exit(errno);
  }}

  get_tc_num_func_t get_dma_num_func =
    (get_tc_num_func_t)dlsym(handle, "get_dma_num");
  if (!get_dma_num_func) {{
    dlclose(handle);
    fprintf(stderr,
            "[ERROR][CMD_GEN] Failed to resolve symbol: `get_matmul_num`, "
            "error: %s", dlerror());
    exit(errno);
  }}

  size_t mm_num = get_matmul_num_func();
  int8_t** h_mm_commands = NULL;
  cl_mem* d_mm_commands = NULL;

  if (mm_num > 0) {{
    generate_matmul_command_func_t generate_matmul_command_func =
        (generate_matmul_command_func_t)dlsym(handle,
                                              "generate_matmul_command");
    if (!generate_matmul_command_func) {{
      dlclose(handle);
      fprintf(stderr,
              "[ERROR][CMD_GEN] Failed to resolve symbol: `generate_matmul_command`, "
              "error: %s", dlerror());
      exit(errno);
    }}

    h_mm_commands = (int8_t**)malloc(mm_num * sizeof(int8_t*));
    d_mm_commands = (cl_mem*)malloc(mm_num * sizeof(cl_mem));

    // TODO(TC): Pass grid_size by values.
    size_t grid_size[3] = {{
        (size_t)grid_dim_x,
        (size_t)grid_dim_y,
        (size_t)grid_dim_z,
    }};

    for (size_t i = 0; i < mm_num; ++i) {{
      h_mm_commands[i] = (int8_t*)malloc(cmd_size_all_blocks);
      if (!generate_matmul_command_func(
          i, grid_size,
          {", ".join(f"arg_{i}" for i in params)},
          grid_dim_x, grid_dim_y, grid_dim_z,
          {", ".join(f"arg_{i}_npy.shape.data()" for i in tensor_params)},
          h_mm_commands[i])) {{
        fprintf(stderr, "Failed to generate the #%zu TC mm command\\n", i);
        exit(1);
      }}

      d_mm_commands[i] = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cmd_size_all_blocks,
        h_mm_commands[i],
        &err
      );
      OCL_CHECK(err);
    }}
  }}

  size_t dma_num = get_dma_num_func();
  int8_t** h_dma_commands = NULL;
  cl_mem* d_dma_commands = NULL;

  if (dma_num > 0) {{
    generate_dma_command_func_t generate_dma_command_func =
      (generate_dma_command_func_t)dlsym(handle, "generate_dma_command");
    if (!generate_dma_command_func) {{
      dlclose(handle);
      fprintf(stderr,
              "[ERROR][CMD_GEN] Failed to resolve symbol: `generate_dma_command`, "
              "error: %s", dlerror());
      exit(errno);
    }}

    h_dma_commands = (int8_t**)malloc(dma_num * sizeof(int8_t*));
    d_dma_commands = (cl_mem*)malloc(dma_num * sizeof(cl_mem));

    for (size_t i = 0; i < dma_num; ++i) {{
      h_dma_commands[i] = (int8_t*)malloc(cmd_size_all_blocks);
      if (!generate_dma_command_func(
          i,
          {", ".join(f"arg_{i}" for i in params)},
          grid_dim_x, grid_dim_y, grid_dim_z,
          {", ".join(f"arg_{i}_npy.shape.data()" for i in tensor_params)},
          h_dma_commands[i])) {{
        fprintf(stderr, "Failed to generate the #%zu TC dma command\\n", i);
        exit(1);
      }}

      d_dma_commands[i] = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cmd_size_all_blocks,
        h_dma_commands[i],
        &err
      );
      OCL_CHECK(err);
    }}
  }}

  _launch(
    grid_dim_x, grid_dim_y, grid_dim_z, num_warps,
    queue, kernel,
    {", ".join(f"d_arg_{i}" for i in tensor_params)},
    {", ".join(f"arg_{i}" for i in scalar_params) + ',' if len(scalar_params) > 0 else ''}
    d_mm_commands, mm_num, d_dma_commands, dma_num
  );

  // Read tensor buffers back to npy.
{
  NEW_LINE.join(
    f"  OCL_CHECK(clEnqueueReadBuffer(queue, d_arg_{i}, true, 0, arg_{i}_size, arg_{i}, 0, nullptr, nullptr));"
    for i in tensor_params
  )
}
  OCL_CHECK(clFinish(queue));

  // Save tensor args to npy files.
{
  NEW_LINE.join(
    f'  npy::write_npy<{_triton_to_c_scalar_type(signature[i])}>("outputs/arg_{i}.npy", arg_{i}_npy);'
    for i in tensor_params
  )
}

  if (mm_num > 0) {{
    for (size_t i = 0; i < mm_num; ++i) {{
        free(h_mm_commands[i]);
        OCL_CHECK(clReleaseMemObject(d_mm_commands[i]));
    }}
    free(h_mm_commands);
    free(d_mm_commands);
  }}

  if (dma_num > 0) {{
    for (size_t i = 0; i < dma_num; ++i) {{
        free(h_dma_commands[i]);
        OCL_CHECK(clReleaseMemObject(d_dma_commands[i]));
    }}
    free(h_dma_commands);
    free(d_dma_commands);
  }}

  // Deinitialize OpenCL.
{
  NEW_LINE.join(
    f"  clReleaseMemObject(d_arg_{i});"
    for i in tensor_params
  )
}

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);

  for (size_t i = 0; i < num_devices; i++) {{
    clReleaseDevice(devices[i]);
  }}
  free(devices);

  clReleaseContext(context);

  printf("[DONE]\\n");
  return 0;
}}
    """
    return src


class VSILauncher(object):

    def __init__(self, src: ASTSource, metadata: NamedTuple):
        # fmt: off
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_index = lambda x: src.fn.arg_names.index(x) if isinstance(x, str) else x
        constants = {arg_index(key): value for key, value in constants.items()}
        signature = {arg_index(key): value for key, value in src.signature.items()}
        # fmt: on
        src = _make_launcher(signature, constants, metadata)

        if metadata.dump_standalone:
            dump_dir = get_dump_dir(metadata.name)
            mod = _compile_module_from_src(src, "__triton_launcher", dump_dir)
            standalone_src = _make_standalone_launcher(signature, constants,
                                                       metadata)
            standalone_dump_path = dump_dir / f"{metadata.name}_standalone.cpp"
            standalone_dump_path.write_text(standalone_src)
            # copy npy.hpp to dump dir
            import shutil
            shutil.copy(_BACKEND_ROOT_DIR / "include/npy.hpp",
                        dump_dir / "npy.hpp")
        else:
            mod = _compile_module_from_src(src, "__triton_launcher")

        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class VSIDriver(DriverBase):

    def __init__(self):
        super().__init__()

        try:
            import torch.vsi

            # TODO(Inductor): Used in inductor autotune, verify on torch v2.7.0.
            self.get_device_interface = lambda: torch.vsi
            self.get_device_capability = torch.vsi.get_device_properties
            self.get_current_stream = torch.vsi._get_current_raw_stream
            self.get_current_device = torch.vsi.current_device
            self.set_current_device = torch.vsi.set_device
        except ImportError:
            raise RuntimeError("vpex is not available!")

        self.utils = VSIUtils()
        self.launcher_cls = VSILauncher

    @staticmethod
    def is_active() -> bool:
        if not hasattr(torch, "vsi"):
            return False
        return torch.vsi.is_available()

    def get_current_target(self) -> GPUTarget:
        prop = self.get_device_interface().get_device_properties()
        # TODO: arch
        return GPUTarget("vsi", 0, prop.warp_size)

    def get_active_torch_device(self) -> torch.device:
        return torch.device("vsi", self.get_current_device())

    def get_benchmarker(self) -> Callable:
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self) -> torch.Tensor:
        cache_size = 256
        return torch.empty(cache_size // 4, dtype=torch.int32, device="vsi")

    def clear_cache(self, cache: torch.Tensor) -> None:
        cache.zero_()