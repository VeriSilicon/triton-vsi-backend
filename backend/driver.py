import os
import hashlib
import tempfile
import functools
from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.runtime.build import _build

# TODO: relative import conflict, fix after wheel ready
# from .compiler import _get_cmdgen_path, _get_dump_dir
def _get_cmdgen_path() -> Path:
    # TODO(): Put libcmdgen.so in tmpdir.
    path = os.getenv("CMD_GEN_PATH")
    if not path:
        raise Exception("CMD_GEN_PATH is not set.")
    return Path(path) / "libcmdgen.so"

def _get_dump_dir() -> Path:
    path = os.getenv("DUMP_DIR")
    if not path:
        raise Exception("DUMP_DIR is not set.")
    return Path(path)

curr_dir = os.path.dirname(os.path.realpath(__file__))

libraries = ["OpenCL"]


@functools.lru_cache()
def vsi_lib_dirs():
    # path to vsi' /sdk/drivers
    driver_path = os.getenv("VSI_DRIVER_PATH")
    if driver_path:
        return [driver_path]
    else:
        raise Exception("VSI_DRIVER_PATH is not set.")


@functools.lru_cache()
def vsi_inc_dirs():
    # path to vsi' /sdk/include
    inc_path = os.getenv("VSI_INCLUDE_PATH")
    if inc_path:
        return [inc_path]
    else:
        raise Exception("VSI_INCLUDE_PATH is not set.")


@functools.lru_cache()
def library_dirs():
    return [*vsi_lib_dirs()]


@functools.lru_cache()
def include_dirs():
    return [*vsi_inc_dirs()]


def compile_module_from_src(src, name, dump_standalone=False):
    if not dump_standalone:
        key = hashlib.sha256(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        cache_path = cache.get_file(f"{name}.so")
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build(name, src_path, tmpdir, library_dirs(),
                            include_dirs(), libraries)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), f"{name}.so", binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    else:
        # Multi kernel may has problems
        src_path = os.path.join(curr_dir, f"{name}.c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, curr_dir, library_dirs(),
                    include_dirs(), libraries)
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


class VSIUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(VSIUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            Path(os.path.join(curr_dir, "driver.c")).read_text(), "vsi_utils")
        self.load_binary = mod.load_binary
        # self.get_device_properties = mod.get_device_properties

    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2 ** 20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None
        }


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "cl_mem"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def ty_convert(ty):
    cpp_ty = ty_to_cpp(ty)
    return cpp_ty if cpp_ty != "cl_mem" else "void *"


def _extracted_type(ty):
    if ty[0] == '*':
        return "PyObject*"
    return ty_to_cpp(ty)


def format_of(ty):
    return {
        "PyObject*": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


def _extract_ptr_type(ty):
    if ty[0] == '*':
        return ty_to_cpp(ty[1:])
    else:
        return "no_ptr"


def make_launcher(constants, signature, metadata, ids):
    arg_decls = ', '.join(
        f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    args_format = ''.join([format_of(_extracted_type(ty))
                          for ty in signature.values()])
    format = "iiiKOOOOO" + args_format
    args_list = ', ' + \
        ', '.join(f"&_arg{i}" for i, ty in signature.items()
                  ) if len(signature) > 0 else ''

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"d_arg{i}")
        else:
            internal_args_list.append(f"_arg{i}")
    quote = "\""
    params = [i for i in signature.keys() if i not in constants]
    src = f"""
#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include "CL/opencl.hpp"
#include <stdbool.h>
#include <Python.h>
#include <numeric>
#include <vector>
#include <dlfcn.h>

static inline void oclAssert(cl_int code, const char *file, int line)
{{
   if (code == CL_SUCCESS)
    return;
  const char* prefix = "Triton Error [OpenCL] ";
  char err[1024] = {{0}};
  snprintf(err, sizeof(err), "%s code %d, %s:%d", prefix, code, file, line);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
}}


static inline void oclAssertMsg(cl_int code, const char* msg, const char *file, int line)
{{
   if (code == CL_SUCCESS)
    return;
  const char* prefix = "Triton Error [OpenCL] ";
  char err[1024] = {{0}};
  snprintf(err, sizeof(err), "%s %s code %d, %s:%d", prefix, msg, code, file, line);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
}}

#define OCL_CHECK(ans) {{ oclAssert((ans), __FILE__, __LINE__); }}
#define OCL_CHECK_MSG(ans, msg) {{ oclAssertMsg((ans), (msg), __FILE__, __LINE__);}}

typedef struct _DevicePtrInfo {{
    cl_mem dev_ptr;
    std::vector<int> shape;
    bool valid;
    size_t num_el;
}} DevicePtrInfo;

static bool getTensorShape(PyObject* obj, std::vector<int> &shape) {{
  PyObject *shape_tuple = PyObject_GetAttrString(obj, "shape");
  if (!shape_tuple) {{
    PyErr_SetString(PyExc_AttributeError, "Object has no shape attribute");
    return false;
  }}
  if (!PyTuple_Check(shape_tuple)) {{
    PyErr_SetString(PyExc_TypeError, "Tensor shape is not a tuple");
    Py_DECREF(shape_tuple);
    return false;
  }}
  int size = PyTuple_Size(shape_tuple);
  if (size > 4) {{
    PyErr_SetString(PyExc_TypeError, "Tensor rank bigger than 4 not support in CL");
    Py_DECREF(shape_tuple);
    return false;
  }}
  for (int i = 0; i < 4; ++i) {{
    if (i > size - 1) {{
      shape.push_back(1);
      continue;
    }}
    PyObject* dim = PyTuple_GetItem(shape_tuple, i);
    if (!PyLong_Check(dim)) {{
      PyErr_SetString(PyExc_TypeError, "Tensor shape dimensions must be integers");
      Py_DECREF(shape_tuple);
      return false;
    }}
    shape.push_back(PyLong_AsLong(dim));
  }}
  Py_DECREF(shape_tuple);
  return true;
}}

static DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info = {{NULL, std::vector<int>(), true, 0}};
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}

  PyObject *ptr_method = PyObject_GetAttrString(obj, "data_ptr");
  if(!ptr_method){{
    PyErr_SetString(PyExc_AttributeError, "Object has no data_ptr attribute");
    return ptr_info;
  }}
  PyObject *empty_tuple = PyTuple_New(0);
  PyObject *ret = PyObject_Call(ptr_method, empty_tuple, NULL);
  Py_DECREF(empty_tuple);
  Py_DECREF(ptr_method);
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    return ptr_info;
  }}
  cl::Buffer *buffer = (cl::Buffer*)PyLong_AsVoidPtr(ret);
  ptr_info.dev_ptr = buffer->get();
  if(!ptr_info.dev_ptr) {{
    PyErr_SetString(PyExc_TypeError, "cl_mem of tensor no valid");
    return ptr_info;
  }}
  Py_DECREF(ret);

  if (!getTensorShape(obj, ptr_info.shape)) {{
    return ptr_info;
  }}
  ptr_info.num_el = std::accumulate(ptr_info.shape.begin(), ptr_info.shape.end(), 1, std::multiplies<int>());
  return ptr_info;
}}

// Global OpenCL handles
cl_device_id device;
cl_context context = NULL;
cl_command_queue queue = NULL;

static void initializeOCL(PyObject *cl_tensor) {{
  cl_int err;

  PyObject *ptr_method = PyObject_GetAttrString(cl_tensor, "data_ptr");
  if(!ptr_method) {{
    PyErr_SetString(PyExc_TypeError, "cl tensor do not have data_ptr attribute");
  }}

  PyObject *empty_tuple = PyTuple_New(0);
  PyObject *ret = PyObject_Call(ptr_method, empty_tuple, NULL);
  Py_DECREF(empty_tuple);
  Py_DECREF(ptr_method);
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
  }}

  cl::Buffer *buffer = (cl::Buffer*)PyLong_AsVoidPtr(ret);
  cl_mem handle = buffer->get();
  Py_DECREF(ret);

  err = clGetMemObjectInfo(handle, CL_MEM_CONTEXT, sizeof(cl_context), &context, NULL);
  OCL_CHECK_MSG(err, "Failed to get context from cl_mem");

  size_t device_list_size = 0;
  err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &device_list_size);
  OCL_CHECK_MSG(err, "Failed to get cl device at phase 0");

  cl_device_id* devices = (cl_device_id*)malloc(device_list_size);
  err = clGetContextInfo(context, CL_CONTEXT_DEVICES, device_list_size, devices, NULL);
  OCL_CHECK_MSG(err, "Failed to get cl device at phase 1");
  device = devices[0];

  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  OCL_CHECK_MSG(err, "clCreateCommandQueue failed");
}}

static void deInitializeOCL() {{
  OCL_CHECK_MSG(clReleaseCommandQueue(queue), "clReleaseCommandQueue failed");
}}

static void _launch(int gridX, int gridY, int gridZ,
                    int clusterDimX, int clusterDimY, int clusterDimZ,
                    cl_kernel kernel{', ' + arg_decls if len(arg_decls) > 0 else ''},
                    cl_mem d_commands) {{

  size_t arg_count = 0;
  // set raw kernel args
{chr(10).join([f"  OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof({ty_to_cpp(signature[i])}), &arg{i})); arg_count++;" for i in params])}

  // set extra kernel args
  OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(int), &gridX));
  OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(int), &gridY));
  OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(int), &gridZ));

  // set tc commands arg
  if (d_commands)
    OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &d_commands));

  if (gridX*gridY*gridZ > 0) {{
    size_t global_size[] = {{(size_t)gridX * clusterDimX, (size_t)gridY * clusterDimY, (size_t)gridZ * clusterDimZ}};
    size_t local_size[] = {{(size_t)clusterDimX, (size_t)clusterDimY, (size_t)clusterDimZ}};
    OCL_CHECK_MSG(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size,
                                         local_size, 0, NULL, NULL),
                  "clEnqueueNDRangeKernel failed");
    OCL_CHECK(clFinish(queue));
    OCL_CHECK(clReleaseKernel(kernel));
  }} else {{
    PyErr_SetString(PyExc_RuntimeError, "Global work size error");
  }}

}}

extern "C" {{
typedef int (*get_matmul_num_ptr)(void); // libcmdgen api
// declaration libcmdgen api
// The parameters are:                      idx, grids,  kernel_args...                                          gX,  gY,  gZ, tensor_shapes...
typedef bool (*generate_matmul_command_ptr)(int, void *, {', '.join(ty_convert(signature[i]) for i in params)}, int, int, int, {', '.join("void *" for i in params if ty_to_cpp(signature[i]) == "cl_mem")}, void *);

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  cl_int err = CL_SUCCESS;
  char* data;
  PyObject* binary = NULL;
  Py_ssize_t data_size;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &binary,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  initializeOCL({f"_arg{next(i for i, ty in signature.items() if ty[0] == '*')}"});

  PyObject* data_obj = PyDict_GetItemString(binary, "data");
  if (PyBytes_AsStringAndSize(data_obj, &data, &data_size) != 0) {{
    return NULL;
  }}

  // Get Spirv kernel
  const char* spirv_data = data;
  size_t spirv_size = data_size;

  // Get kernel name
  const char* kernel_name = "{metadata.name}";

  // Load kernel
  cl_program prog = clCreateProgramWithIL(context, spirv_data, spirv_size, &err);
  OCL_CHECK_MSG(err, "clCreateProgramWithIL failed");
  OCL_CHECK_MSG(clBuildProgram(prog, 1, &device, NULL, NULL, NULL),
                "clBuildProgram failed");

  cl_kernel kernel = clCreateKernel(prog, kernel_name, &err);
  OCL_CHECK_MSG(err, "clCreateKernel failed");
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

{chr(10).join(f"  DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" for i, ty in signature.items() if ty[0] == "*")}
{chr(10).join(f"  cl_mem d_arg{i} = ptr_info{i}.dev_ptr;" for i, ty in signature.items() if ty[0] == "*")}

  // Generate TC commands

  void *handle = dlopen({quote}{_get_cmdgen_path()}{quote}, RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_OSError, "can not load libcmdgen.so");
    return NULL;
  }}
  get_matmul_num_ptr get_matmul_num = (get_matmul_num_ptr)dlsym(handle, "get_matmul_num");
  if (!get_matmul_num) {{
    dlclose(handle);
    PyErr_SetString(PyExc_OSError, "undefined symbol get_matmul_num in libcmdgen.so");
    return NULL;
  }}
  int cmd_num = get_matmul_num();

  generate_matmul_command_ptr generate_matmul_command;
  if (cmd_num != 0) {{
    generate_matmul_command = (generate_matmul_command_ptr)dlsym(handle, "generate_matmul_command");
    if (!generate_matmul_command) {{
      dlclose(handle);
      PyErr_SetString(PyExc_OSError, "undefined symbol generate_matmul_command in libcmdgen.so");
      return NULL;
    }}
  }}

  #define CMD_SIZE 256
  int grids_num = gridX * gridY * gridZ;
  int step = grids_num * CMD_SIZE;
  int cmds_size = CMD_SIZE * cmd_num * grids_num;
  void* h_commands = cmd_num == 0 ? NULL : malloc(cmds_size);
  int grids[3] = {{gridX, gridY, gridZ}};
  for (int i = 0; i < cmd_num; ++i) {{
    if (!generate_matmul_command(i, grids, {', '.join(f"_arg{i}" for i in params)}, gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.shape.data()" for i, ty in signature.items() if ty[0] == "*")}, (char*)h_commands + step * i))
      fprintf(stderr, "Generate %dth TC command failed\\n", i);
  }}

  cl_mem d_commands = NULL;
  if (cmd_num != 0) {{
    d_commands = clCreateBuffer(context, CL_MEM_READ_ONLY, cmds_size, NULL, &err);
    OCL_CHECK_MSG(err, "clCreateBuffer for commands failed");
    err = clEnqueueWriteBuffer(queue, d_commands, CL_TRUE, 0, cmds_size, h_commands, 0, NULL, NULL);
    OCL_CHECK_MSG(err, "clEnqueueWriteBuffer for commands failed");
  }}

  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, clusterDimX, clusterDimY, clusterDimZ, kernel, {', '.join(internal_args_list) if len(internal_args_list) > 0 else ''}, d_commands);
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred())
    return NULL;


  // Free
  if (d_commands)
    OCL_CHECK(clReleaseMemObject(d_commands));
  if (h_commands)
    free(h_commands);

  deInitializeOCL();

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
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

}} // extern "C"
"""
    return src


def make_standalone_launcher(constants, signature, metadata, ids, dump_dir):
    arg_decls = ', '.join(
        f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    internal_args_list = []
    tensor_npys = {}
    other_npys = {}
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"d_arg{i}")
            tensor_npys[i] = f"_arg{i}.npy"
        else:
            internal_args_list.append(f"_arg{i}")
            other_npys[i] = f"_arg{i}.npy"

    params = [i for i in signature.keys() if i not in constants]

    pack_metadata = {
        "clusterDimX": metadata.cluster_dims[0],
        "clusterDimY": metadata.cluster_dims[1],
        "clusterDimZ": metadata.cluster_dims[2]
    }
    quote = "\""
    src = f"""
#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include "./npy.hpp"
#include <stdbool.h>
#include <fstream>
#include <string>

#ifndef NO_TC_COMMANDS
#include <dlfcn.h>
#endif // NO_TC_COMMANDS

static inline void oclAssert(cl_int code, const char *file, int line)
{{
   if (code == CL_SUCCESS)
    return;
  const char* prefix = "Triton Error [OpenCL] ";
  char err[1024] = {{0}};
  snprintf(err, sizeof(err), "%s code %d, %s:%d", prefix, code, file, line);
  fprintf(stderr, "%s\\n", err);
  puts(""); // as change newline
}}

static inline void oclAssertMsg(cl_int code, const char* msg, const char *file, int line)
{{
   if (code == CL_SUCCESS)
    return;
  const char* prefix = "Triton Error [OpenCL] ";
  char err[1024] = {{0}};
  snprintf(err, sizeof(err), "%s %s code %d, %s:%d", prefix, msg, code, file, line);
  fprintf(stderr, "%s\\n", err);
  puts(""); // as change newline
}}

#define OCL_CHECK(ans) {{ oclAssert((ans), __FILE__, __LINE__); }}
#define OCL_CHECK_MSG(ans, msg) {{ oclAssertMsg((ans), (msg), __FILE__, __LINE__);}}

// Global OpenCL handles
cl_device_id device;
cl_context context = NULL;
cl_command_queue queue = NULL;

static void initializeOCL() {{
  cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
  cl_uint chosen_platform_index = 0;
  cl_uint chosen_device_index = 0;
  cl_uint num_platforms = 0;
  cl_uint num_devices = 0;
  cl_platform_id *platforms;
  cl_device_id *devices = NULL;
  cl_int err;

  OCL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));

  platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));

  OCL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));

  OCL_CHECK(clGetDeviceIDs(platforms[chosen_platform_index], gDeviceType, 0,
                           NULL, &num_devices));

  devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

  OCL_CHECK(clGetDeviceIDs(platforms[chosen_platform_index], gDeviceType,
                           num_devices, devices, NULL));

  device = devices[chosen_device_index];

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  OCL_CHECK_MSG(err, "clCreateContext failed");

  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  OCL_CHECK_MSG(err, "clCreateCommandQueue failed");
}}

static void deInitializeOCL() {{
  OCL_CHECK_MSG(clReleaseCommandQueue(queue), "clReleaseCommandQueue failed");
  OCL_CHECK(clReleaseContext(context));
  OCL_CHECK(clReleaseDevice(device));
}}

static void _launch(int gridX, int gridY, int gridZ,
                    int clusterDimX, int clusterDimY, int clusterDimZ,
                    cl_kernel kernel{', ' + arg_decls if len(arg_decls) > 0 else ''},
                    cl_mem d_commands) {{

  size_t arg_count = 0;
  // set raw kernel args
{chr(10).join(f"  OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof({ty_to_cpp(signature[i])}), &arg{i})); arg_count++;" for i in params)}

  // set extra kernel args
  OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(int), &gridX));
  OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(int), &gridY));
  OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(int), &gridZ));

  // set tc commands arg
  if (d_commands)
    OCL_CHECK(clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &d_commands));

  if (gridX*gridY*gridZ > 0) {{
    size_t global_size[] = {{(size_t)gridX * clusterDimX, (size_t)gridY * clusterDimY, (size_t)gridZ * clusterDimZ}};
    size_t local_size[] = {{(size_t)clusterDimX, (size_t)clusterDimY, (size_t)clusterDimZ}};
    OCL_CHECK_MSG(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size,
                                         local_size, 0, NULL, NULL),
                  "clEnqueueNDRangeKernel failed");
    OCL_CHECK(clFinish(queue));
    OCL_CHECK(clReleaseKernel(kernel));
  }} else {{
    fprintf(stderr, "Global work size error\\n");
    puts(""); // as change newline
  }}

}}

typedef int (*get_matmul_num_ptr)(void); // libcmdgen api
// declaration libcmdgen api
// The parameters are:                      idx, grids,  kernel_args...                                          gX,  gY,  gZ, tensor_shapes...
typedef bool (*generate_matmul_command_ptr)(int, void *, {', '.join(ty_convert(signature[i]) for i in params)}, int, int, int, {', '.join("void *" for i in params if ty_to_cpp(signature[i]) == "cl_mem")}, void *);

static void paddingShape(std::vector<int64_t> &shape) {{
  // The none standalone code ensure the shape size <= 4
  int size = shape.size();
  for (int i = 0; i < 4 - size; ++i)
    shape.push_back(1);
}}

int main() {{
  std::ifstream file("{metadata.name}.spv");
  std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const char* data = str.c_str();
  size_t data_size = file.tellg();
  file.close();

  initializeOCL();
  cl_int err;

  // Get Spirv kernel
  const char* spirv_data = data;
  size_t spirv_size = data_size;

  // Get kernel name
  const char* kernel_name = "{metadata.name}";

  // Load kernel
  cl_program prog = clCreateProgramWithIL(context, spirv_data, spirv_size, &err);
  OCL_CHECK_MSG(err, "clCreateProgramWithIL failed");
  OCL_CHECK_MSG(clBuildProgram(prog, 1, &device, NULL, NULL, NULL),
                "clBuildProgram failed");

  cl_kernel kernel = clCreateKernel(prog, kernel_name, &err);
  OCL_CHECK_MSG(err, "clCreateKernel failed");

  // get grids
  auto grids_npy = npy::read_npy<int32_t>("grid.npy");
  int *grids = grids_npy.data.data();
  int32_t gridX = grids[0];
  int32_t gridY = grids[1];
  int32_t gridZ = grids[2];

{chr(10).join(f"  int32_t {k} = {v};"  for k, v in pack_metadata.items())}

  // get pointer parameters
{chr(10).join(f"  auto _arg{i}_npy = npy::read_npy<{_extract_ptr_type(signature[i])}>({quote}{addr}{quote});" for i, addr in tensor_npys.items())}

{chr(10).join(f"  void* _arg{i} = _arg{i}_npy.data.data();" for i, addr in tensor_npys.items())}

{chr(10).join(f"  size_t arg{i}_size = _arg{i}_npy.data.size();" for i, addr in tensor_npys.items())}

  // get other parameters
{chr(10).join(f"  auto _arg{i}_npy = npy::read_npy<{_extracted_type(signature[i])}>({quote}{addr}{quote});" for i, addr in other_npys.items())}

{chr(10).join(f"  auto _arg{i} = _arg{i}_npy.data.front(); " for i, ty in signature.items() if ty[0] != "*")}

  // Copy args host to device
{chr(10).join(f"  cl_mem d_arg{i} = clCreateBuffer(context, CL_MEM_READ_WRITE, arg{i}_size, NULL, &err); OCL_CHECK(err);" for i, ty in signature.items() if ty[0] == "*")}

{chr(10).join(f"  OCL_CHECK(clEnqueueWriteBuffer(queue, d_arg{i}, CL_TRUE, 0, arg{i}_size, _arg{i}, 0, NULL, NULL));" for i, ty in signature.items() if ty[0] == "*")}


  int cmd_num = 0;
#ifndef NO_TC_COMMANDS
  // Get TC command
  void *handle = dlopen({quote}{_get_cmdgen_path()}{quote}, RTLD_LAZY);
  if (!handle) {{
    fprintf(stderr, "can not load libcmdgen.so\\n");
    return -1;
  }}
  get_matmul_num_ptr get_matmul_num = (get_matmul_num_ptr)dlsym(handle, "get_matmul_num");
  if (!get_matmul_num) {{
    dlclose(handle);
    fprintf(stderr, "undefined symbol get_matmul_num in libcmdgen.so\\n");
    return -1;
  }}
  cmd_num = get_matmul_num();

  generate_matmul_command_ptr generate_matmul_command;
  if (cmd_num != 0) {{
    generate_matmul_command = (generate_matmul_command_ptr)dlsym(handle, "generate_matmul_command");
    if (!generate_matmul_command) {{
      dlclose(handle);
      fprintf(stderr, "undefined symbol generate_matmul_command in libcmdgen.so\\n");
      return -1;
    }}
  }}
#endif // NO_TC_COMMANDS

  #define CMD_SIZE 256
  int grids_num = gridX * gridY * gridZ;
  int step = grids_num * CMD_SIZE;
  int cmds_size = CMD_SIZE * cmd_num * grids_num;
  void* h_commands = cmd_num == 0 ? NULL : malloc(cmds_size);

{chr(10).join(f"  std::vector<int64_t> shape{i}(_arg{i}_npy.shape.begin(), _arg{i}_npy.shape.end()); paddingShape(shape{i});" for i, _ in tensor_npys.items())}

#ifndef NO_TC_COMMANDS
  int64_t c_grids[3] = {{gridX, gridY, gridZ}};
  for (int i = 0; i < cmd_num; ++i) {{
    if (!generate_matmul_command(i, c_grids, {', '.join(f"_arg{i}" for i in params)}, gridX, gridY, gridZ, {', '.join(f"shape{i}.data()" for i, _ in tensor_npys.items())}, (int8_t*)h_commands + step * i))
      fprintf(stderr, "Generate %dth TC command failed\\n", i);
  }}
#endif // NO_TC_COMMANDS

  cl_mem d_commands = NULL;
  if (cmd_num != 0) {{
    d_commands = clCreateBuffer(context, CL_MEM_READ_ONLY, cmds_size, NULL, &err);
    OCL_CHECK_MSG(err, "clCreateBuffer for commands failed");
    err = clEnqueueWriteBuffer(queue, d_commands, CL_TRUE, 0, cmds_size, h_commands, 0, NULL, NULL);
    OCL_CHECK_MSG(err, "clEnqueueWriteBuffer for commands failed");
  }}

  _launch(gridX, gridY, gridZ, clusterDimX, clusterDimY, clusterDimZ, kernel, {', '.join(internal_args_list) if len(internal_args_list) > 0 else ''}, d_commands);

  // Copy result device to host
{chr(10).join(f"  OCL_CHECK(clEnqueueReadBuffer(queue, d_arg{i}, CL_TRUE, 0, arg{i}_size, _arg{i}, 0, NULL, NULL));" for i, ty in signature.items() if ty[0] == "*")}

{chr(10).join(f"  npy::npy_data_ptr<{_extract_ptr_type(ty)}> o{i};" for i, ty in signature.items() if ty[0] == "*")}
{chr(10).join(f"  o{i}.data_ptr = ({_extract_ptr_type(ty)}*)_arg{i};" for i, ty in signature.items() if ty[0] == "*")}
{chr(10).join(f"  o{i}.shape = _arg{i}_npy.shape;" for i, ty in signature.items() if ty[0] == "*")}
{chr(10).join(f"  npy::write_npy({quote}out{i}.npy{quote}, o{i});" for i, ty in signature.items() if ty[0] == "*")}

  // Free
  if (d_commands)
    OCL_CHECK(clReleaseMemObject(d_commands));
  if (h_commands)
    free(h_commands);

{chr(10).join(f"  OCL_CHECK(clReleaseMemObject(d_arg{i}));" for i, ty in signature.items() if ty[0] == "*")}

  deInitializeOCL();

  return 0;
}}

"""
    return src


class VSILauncher(object):
    def __init__(self, src, metadata):

        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(
            src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        def cst_key(i): return src.fn.arg_names.index(
            i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key,
                     value in src.signature.items()}
        src = make_launcher(constants, signature, metadata, ids)
        mod = compile_module_from_src(
            src, "__triton_launcher", metadata.dump_standalone)
        if metadata.dump_standalone:
            dump_dir = _get_dump_dir()
            dump_dir.mkdir(parents=True, exist_ok=True)
            src = make_standalone_launcher(
                constants, signature, metadata, ids, str(dump_dir))
            name = metadata.name
            standalone_dump_path = dump_dir / f"{name}.cpp"
            standalone_dump_path.write_text(src)
            # copy npy.hpp to dump dir
            import shutil
            shutil.copy(curr_dir + "/npy.hpp", dump_dir / "npy.hpp")

        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class VSIDriver(DriverBase):
    def __init__(self):
        super().__init__()
        self.utils = VSIUtils()
        self.launcher_cls = VSILauncher

    def get_device_interface(self):
        import torch
        return torch.vsi

    # VSI driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(VSIDriver())
    @staticmethod
    def is_active():
        return False

    def get_device_capability(self):
        return ("vsi", 0)

    def get_current_stream(self, device):
        return 0

    def get_current_device(self):
        return 0  # 'vsi

    def set_current_device(self, device):
        assert device == 'vsi'
        return 'vsi'

    def get_current_target(self):
        warp_size = 64
        # TODO: arch
        return GPUTarget("vsi", 0, warp_size)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="vsi")
