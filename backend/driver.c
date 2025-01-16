#include <CL/cl.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool oclAssert(cl_int code, const char *file, int line) {
  if (code == CL_SUCCESS)
    return true;
  const char *prefix = "Triton Error [OpenCL]";
  char err[1024] = {0};
  snprintf(err, sizeof(err), "%s code %d, %s:%d", prefix, code, file ,line);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

#define OCL_CHECK(ans)                                                         \
  { oclAssert((ans), __FILE__, __LINE__); }

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define OCL_CHECK_AND_RETURN_NULL(ans)                                         \
  do {                                                                         \
    if (!oclAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define OCL_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                           \
  do {                                                                         \
    if (!oclAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

#define log_info printf
#define log_error printf

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  cl_device_type gDeviceType = CL_DEVICE_TYPE_GPU;
  cl_uint choosen_platform_index = 0;
  cl_uint choosen_device_index = 0;
  cl_uint num_platforms = 0;
  cl_uint num_devices = 0;
  cl_platform_id *platforms;
  cl_device_id device;
  cl_device_id *devices = NULL;
  cl_int err;

  OCL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));

  platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));

  OCL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));

  OCL_CHECK(clGetDeviceIDs(platforms[choosen_platform_index], gDeviceType, 0,
                           NULL, &num_devices));

  devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

  OCL_CHECK(clGetDeviceIDs(platforms[choosen_platform_index], gDeviceType,
                           num_devices, devices, NULL));

  device = devices[choosen_device_index];

  int max_shared_mem;
  int max_num_regs = 0;
  int multiprocessor_count = 8;
  int warp_size = 1;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;

  OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                            &max_shared_mem, NULL));
  OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_ulong), &sm_clock_rate, NULL));
  OCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_ulong),
                            &mem_bus_width, NULL));

  mem_clock_rate = sm_clock_rate;

  free(platforms);
  free(devices);

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }

  return Py_BuildValue("(K{s:y#,s:K}ii)", 0, "data", data,
                       data_size, "size", data_size, 0, 0);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS, "Load provided spirv binary"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "vsi_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_vsi_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
