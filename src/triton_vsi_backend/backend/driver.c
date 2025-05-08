#include <stddef.h>
#include <stdint.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

static void oclAssert(cl_int code, const char* file, int line) {
  if (code == CL_SUCCESS) {
    return;
  }

  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();

  PyErr_Format(PyExc_RuntimeError, "Triton Error [OpenCL]: code: %d, at %s:%d",
               code, file, line);

  PyGILState_Release(gil_state);
}

#define OCL_CHECK(ans)                    \
  {                                       \
    oclAssert((ans), __FILE__, __LINE__); \
    if (PyErr_Occurred()) {               \
      return NULL;                        \
    }                                     \
  }

static PyObject* loadBinary(PyObject* self, PyObject* args) {
  uint64_t raw_context;
  uint64_t raw_device;
  const char* kernel_name;
  const uint8_t* data;
  Py_ssize_t data_size;
  int shared;
  if (!PyArg_ParseTuple(args, "KKss#i", &raw_context, &raw_device, &kernel_name,
                        &data, &data_size, &shared)) {
    return NULL;
  }

  cl_int error;

  cl_context context = (cl_context)(raw_context);
  cl_device_id device = (cl_device_id)(raw_device);

  cl_program program = clCreateProgramWithIL(context, data, data_size, &error);
  OCL_CHECK(error);

  OCL_CHECK(clBuildProgram(program, 1, &device, NULL, NULL, NULL));

  cl_kernel kernel = clCreateKernel(program, kernel_name, &error);
  OCL_CHECK(error);

  int num_regs = 0;
  int num_spills = 0;
  return Py_BuildValue("(KKii)", (uint64_t)(program), (uint64_t)(kernel),
                       num_regs, num_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided SPIR-V binary into OpenCL driver"},
    {NULL, NULL, 0, NULL}  // sentinel.
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "vsi_utils",
                                       NULL,  // documentation
                                       -1,    // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_vsi_utils(void) {
  PyObject* m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
