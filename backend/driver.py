import hashlib
import tempfile
import os
import tempfile

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
import functools
from triton.runtime.build import _build
dirname = os.path.dirname(os.path.realpath(__file__))

libraries = ['OpenCL']

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


def compile_module_from_src(src, name, save_local=False):
    if not save_local:
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
        src_path = os.path.join(dirname, f"{name}.c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, dirname, library_dirs(),
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
            Path(os.path.join(dirname, "driver.c")).read_text(), "vsi_utils")
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

    params = [i for i in signature.keys() if i not in constants]
    EXTRA_ARGS_NUM = 3  # blocksize XYZ
    src = f"""
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <stdbool.h>
#include <Python.h>

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
    bool valid;
    size_t num_el;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info = {{NULL, true, 0}};
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}

  PyObject *ptr_method = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr_method){{
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
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);
  }}

  PyObject *numel_method = PyObject_GetAttrString(obj, "numel");
  if (numel_method) {{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(numel_method, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(numel_method);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "numel method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.num_el = PyLong_AsSize_t(ret);
    Py_DECREF(ret);
  }}
  if (!ptr_info.dev_ptr) {{
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
  }}
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
                    cl_mem* d_commands, int N) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)}, &gridX, &gridY, &gridZ }};
  const size_t original_arg_count = {len(params) + EXTRA_ARGS_NUM};
  void** new_args = (void**)malloc((original_arg_count + N)*sizeof(void*));
  for (int i = 0; i < N ; i++) {{
    new_args[i] = (void*)&d_commands[i];
  }}
  memcpy(new_args + N, params, original_arg_count*sizeof(void*));

  size_t arg_count = 0;
  // set commands args
  while (arg_count < N) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof(cl_mem), new_args[arg_count]));
    arg_count++;
  }}

  // set raw kernel args
{chr(10).join([f"  OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof({ty_to_cpp(signature[i])}), new_args[arg_count])); arg_count++;" for i in params])}

  // set extra kernel args
  while(arg_count < original_arg_count + N) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof(int), new_args[arg_count]));
    arg_count++;
  }}

  if (gridX*gridY*gridZ > 0) {{
    // Currently VSI not handle multiple local threads
    size_t global_size[] = {{(size_t)gridX, (size_t)gridY, (size_t)gridZ}};
    size_t local_size[] = {{1, 1, 1}};
    OCL_CHECK_MSG(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size,
                                         local_size, 0, NULL, NULL),
                  "clEnqueueNDRangeKernel failed");
    OCL_CHECK(clFinish(queue));
    OCL_CHECK(clReleaseKernel(kernel));
  }} else {{
    PyErr_SetString(PyExc_RuntimeError, "Global work size error");
  }}

  free(new_args);
}}

extern "C" {{

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
  // Get TC command
  #define CMD_SIZE 256
  int N = (int)(unsigned char)data[0];
  size_t commands_size = N * CMD_SIZE;
  cl_mem* d_commands = (cl_mem*)malloc(N * sizeof(cl_mem*));
  for (int i = 0; i < N; i ++){{
    d_commands[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, CMD_SIZE, NULL, &err);
    OCL_CHECK_MSG(err, "clCreateBuffer for commands failed");
    err = clEnqueueWriteBuffer(queue, d_commands[i], CL_TRUE, 0, CMD_SIZE, data+1+i*CMD_SIZE, 0, NULL, NULL);
    OCL_CHECK_MSG(err, "clEnqueueWriteBuffer for commands failed");
  }}

  // Get Spirv kernel
  const char* spirv_data = data + 1 + commands_size;
  size_t spirv_size = data_size - 1 - commands_size;
  FILE* spv_file = fopen("temp.spv", "wb"); // save for debug
  fwrite(spirv_data, 1, spirv_size, spv_file);
  fclose(spv_file);

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

  // raise exception asap
{chr(10).join([f"  DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])}
{chr(10).join([f"  cl_mem d_arg{i} = ptr_info{i}.dev_ptr;" if ty[0] == "*" else "" for i, ty in signature.items()])}

  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, clusterDimX, clusterDimY, clusterDimZ, kernel{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''}, d_commands, N);
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred())
    return NULL;


  // Free
  for (int i = 0; i < N; i ++){{
    OCL_CHECK(clReleaseMemObject(d_commands[i]));
  }}
  if (d_commands)
    free(d_commands);

  deInitializeOCL();

//  if (remove("temp.spv") != 0)
//    return NULL;

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

def make_standalone_launcher(constants, signature, metadata, ids):
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
    EXTRA_ARGS_NUM = 3  # blocksize XYZ

    pack_metadata = {
        "clusterDimX": metadata.cluster_dims[0],
        "clusterDimY": metadata.cluster_dims[1],
        "clusterDimZ": metadata.cluster_dims[2]
    }
    quote = "\""
    src = f"""
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <stdbool.h>
#include <cnpy.h>
#include <fstream>
#include <string>


static inline void oclAssert(cl_int code, const char *file, int line)
{{
   if (code == CL_SUCCESS)
    return;
  const char* prefix = "Triton Error [OpenCL] ";
  char err[1024] = {{0}};
  snprintf(err, sizeof(err), "%s code %d, %s:%d", prefix, code, file, line);
  fprintf(stderr, "%s", err);
  puts(""); // as change newline
}}

static inline void oclAssertMsg(cl_int code, const char* msg, const char *file, int line)
{{
   if (code == CL_SUCCESS)
    return;
  const char* prefix = "Triton Error [OpenCL] ";
  char err[1024] = {{0}};
  snprintf(err, sizeof(err), "%s %s code %d, %s:%d", prefix, msg, code, file, line);
  fprintf(stderr, "%s", err);
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
                    cl_mem* d_commands, int N) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)}, &gridX, &gridY, &gridZ }};
  const size_t original_arg_count = {len(params) + EXTRA_ARGS_NUM};
  void** new_args = (void**)malloc((original_arg_count + N)*sizeof(void*));
  for (int i = 0; i < N ; i++) {{
    new_args[i] = (void*)&d_commands[i];
  }}
  memcpy(new_args + N, params, original_arg_count*sizeof(void*));

  size_t arg_count = 0;
  // set commands args
  while (arg_count < N) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof(cl_mem), new_args[arg_count]));
    arg_count++;
  }}

  // set raw kernel args
{chr(10).join([f"  OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof({ty_to_cpp(signature[i])}), new_args[arg_count])); arg_count++;" for i in params])}

  // set extra kernel args
  while(arg_count < original_arg_count + N) {{
    OCL_CHECK(clSetKernelArg(kernel, arg_count, sizeof(int), new_args[arg_count]));
    arg_count++;
  }}

  if (gridX*gridY*gridZ > 0) {{
    // Currently VSI not handle multiple local threads
    size_t global_size[] = {{(size_t)gridX, (size_t)gridY, (size_t)gridZ}};
    size_t local_size[] = {{1, 1, 1}};
    OCL_CHECK_MSG(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size,
                                         local_size, 0, NULL, NULL),
                  "clEnqueueNDRangeKernel failed");
    OCL_CHECK(clFinish(queue));
    OCL_CHECK(clReleaseKernel(kernel));
  }} else {{
    fprintf(stderr, "Global work size error");
    puts(""); // as change newline
  }}

  free(new_args);
}}

int main() {{
  // get grids
  auto grids_npy = cnpy::npy_load("grid.npy");
  auto grids = grids_npy.as_vec<int32_t>();
  int32_t gridX = grids[0];
  int32_t gridY = grids[1];
  int32_t gridZ = grids[2];

  std::ifstream file("{metadata.name}.spv");
  std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const char* data = str.c_str();
  size_t data_size = file.tellg();
  file.close();

  initializeOCL();
  cl_int err;

  // Get TC command
  #define CMD_SIZE 256
  int N = (int)(unsigned char)data[0];
  size_t commands_size = N * CMD_SIZE;
  cl_mem* d_commands = (cl_mem*)malloc(N * sizeof(cl_mem*));
  for (int i = 0; i < N; i ++){{
    d_commands[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, CMD_SIZE, NULL, &err);
    OCL_CHECK_MSG(err, "clCreateBuffer for commands failed");
    err = clEnqueueWriteBuffer(queue, d_commands[i], CL_TRUE, 0, CMD_SIZE, data+1+i*CMD_SIZE, 0, NULL, NULL);
    OCL_CHECK_MSG(err, "clEnqueueWriteBuffer for commands failed");
  }}

  // Get Spirv kernel
  const char* spirv_data = data + 1 + commands_size;
  size_t spirv_size = data_size - 1 - commands_size;
  FILE* spv_file = fopen("temp.spv", "wb"); // save for debug
  fwrite(spirv_data, 1, spirv_size, spv_file);
  fclose(spv_file);

  // Get kernel name
  const char* kernel_name = "{metadata.name}";

  // Load kernel
  cl_program prog = clCreateProgramWithIL(context, spirv_data, spirv_size, &err);
  OCL_CHECK_MSG(err, "clCreateProgramWithIL failed");
  OCL_CHECK_MSG(clBuildProgram(prog, 1, &device, NULL, NULL, NULL),
                "clBuildProgram failed");

  cl_kernel kernel = clCreateKernel(prog, kernel_name, &err);
  OCL_CHECK_MSG(err, "clCreateKernel failed");

{chr(10).join([f"  int32_t {k} = {v};"  for k, v in pack_metadata.items()])}

  // get pointer parameters
{chr(10).join([f"  auto _arg_{i}_npy = cnpy::npy_load({quote}{addr}{quote});" for i, addr in tensor_npys.items()])}

{chr(10).join([f"  void* _arg{i} = _arg_{i}_npy.data<void>();" for i, addr in tensor_npys.items()])}

{chr(10).join([f"  size_t arg{i}_size = _arg_{i}_npy.num_bytes();" for i, addr in tensor_npys.items()])}

  // get other parameters
{chr(10).join([f"  auto _arg_{i}_npy = cnpy::npy_load({quote}{addr}{quote});" for i, addr in other_npys.items()])}

{chr(10).join([f"  {_extracted_type(ty)} _arg{i} = *(_arg_{i}_npy.data<{_extracted_type(ty)}>()); " if ty[0] != "*" else "" for i, ty in signature.items()])}

  // Copy args host to device
{chr(10).join([f"  cl_mem d_arg{i} = clCreateBuffer(context, CL_MEM_READ_WRITE, arg{i}_size, NULL, &err); OCL_CHECK(err);" if ty[0] == "*" else "" for i, ty in signature.items()])}

{chr(10).join([f"  OCL_CHECK(clEnqueueWriteBuffer(queue, d_arg{i}, CL_TRUE, 0, arg{i}_size, _arg{i}, 0, NULL, NULL));" if ty[0] == "*" else "" for i, ty in signature.items()])}

  _launch(gridX, gridY, gridZ, clusterDimX, clusterDimY, clusterDimZ, kernel{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''}, d_commands, N);

  // Copy result device to host
{chr(10).join([f"  OCL_CHECK(clEnqueueReadBuffer(queue, d_arg{i}, CL_TRUE, 0, arg{i}_size, _arg{i}, 0, NULL, NULL));" if ty[0] == "*" else "" for i, ty in signature.items()])}

{chr(10).join([f"  cnpy::npy_save({quote}out_arg{i}.npy{quote}, ({_extract_ptr_type(ty)}*)_arg{i}, _arg_{i}_npy.shape); " if ty[0] == "*" else "" for i, ty in signature.items()])}

  // Free
  for (int i = 0; i < N; i ++){{
    OCL_CHECK(clReleaseMemObject(d_commands[i]));
  }}
  if (d_commands)
    free(d_commands);

{chr(10).join([f"  OCL_CHECK(clReleaseMemObject(d_arg{i}));" if ty[0] == "*" else "" for i, ty in signature.items()])}

  deInitializeOCL();

  if (remove("temp.spv") != 0)
    return -1;
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
            src = make_standalone_launcher(
                constants, signature, metadata, ids)
            name = metadata.name
            if name and not os.path.exists(name):
                os.makedirs(name)
            src_path = os.path.join(name, f"{name}.cpp")
            with open(src_path, "w") as f:
                f.write(src)

        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)

class VSIDriver(DriverBase):
    def __init__(self):
        super().__init__()
        self.utils = VSIUtils()
        self.launcher_cls = VSILauncher
        self.binary_ext = "spv"

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

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
