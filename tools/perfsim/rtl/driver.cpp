// Isolated Verilator DPI shim. No Ramulator algorithms or RTL datapath are duplicated here.
#include <Python.h>
#include <svdpi.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
static PyObject* driver=nullptr;
static void check(bool ok) { if(!ok){PyErr_Print();std::fputs("perfsim DPI Python failure\n",stderr);std::abort();} }
extern "C" void pm_init(const char* profile,const char* directory,const char* root,const char* site,const char* executable,int width,int words){
  PyConfig config;PyConfig_InitPythonConfig(&config);
  check(!PyStatus_Exception(PyConfig_SetBytesString(&config,&config.program_name,executable)));
  PyStatus status=Py_InitializeFromConfig(&config);PyConfig_Clear(&config);check(!PyStatus_Exception(status));
  PyObject* path=PySys_GetObject("path");
  for(auto value:{root,site}){PyObject* s=PyUnicode_FromString(value);check(PyList_Insert(path,0,s)==0);Py_DECREF(s);}
  PyObject* mod=PyImport_ImportModule("tools.perfsim.rtl.memory_driver");check(mod);
  PyObject* cls=PyObject_GetAttrString(mod,"Driver");check(cls);
  driver=PyObject_CallFunction(cls,"ssii",profile,directory,width,words);check(driver);Py_DECREF(cls);Py_DECREF(mod);
}
static PyObject* desc(int valid,int write,unsigned long long addr,unsigned long long mask,int tag,const void* data,int size){
  return Py_BuildValue("(iiKKiN)",valid,write,addr,mask,tag,PyBytes_FromStringAndSize(static_cast<const char*>(data),size));
}
static int integer(PyObject* tuple,int idx){return static_cast<int>(PyLong_AsLong(PyTuple_GetItem(tuple,idx)));}
static void bytes(PyObject* tuple,int idx,void* out,int capacity){PyObject* b=PyTuple_GetItem(tuple,idx);check(PyBytes_Check(b));auto n=PyBytes_Size(b);check(n<=capacity);std::memset(out,0,capacity);std::memcpy(out,PyBytes_AsString(b),n);}
extern "C" void pm_step(long long cycle,int ar,int aw,unsigned int aa,unsigned int av,unsigned int am,
 int br,int bw,unsigned int ba,unsigned int bm,int bt,const svBitVecVal* bd,
 int sr,unsigned int sa,unsigned int sv,unsigned int sm,
 int* ready,int* arv,svBitVecVal* ard,int* brv,int* brt,svBitVecVal* brd,int* ri,int* wi){
  PyObject* a=desc(ar,aw,4ull*aa,am,0,&av,4);
  PyObject* b=desc(br,bw,4ull*ba,bm,bt,bd,128);
  PyObject* s=desc(sr,1,4ull*sa,sm,0,&sv,4);
  PyObject* result=PyObject_CallMethod(driver,"step","LNNN",cycle,a,b,s);check(result);
  *ready=integer(result,0);*arv=integer(result,1);bytes(result,2,ard,32);
  *brv=integer(result,3);*brt=integer(result,4);bytes(result,5,brd,128);
  *ri=integer(result,6);*wi=integer(result,7);Py_DECREF(result);
}
extern "C" void pm_dump(){PyObject* r=PyObject_CallMethod(driver,"dump",nullptr);check(r);Py_DECREF(r);}

extern "C" void pm_finish(){if(driver){PyObject* r=PyObject_CallMethod(driver,"close",nullptr);check(r);Py_DECREF(r);}}
