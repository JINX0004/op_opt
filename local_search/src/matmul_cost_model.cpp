#include <Python.h>
#include "matmul_cost_model.hpp"
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace matmul {
namespace {
// Every owned Python reference is created and destroyed while holding the GIL.
struct Decref { void operator()(PyObject* p) const { Py_XDECREF(p); } };
using Ref = std::unique_ptr<PyObject,Decref>;
void initialize_python() {
    static std::once_flag once;
    std::call_once(once,[] {
        if(!Py_IsInitialized()) {
            Py_Initialize();
            PyEval_SaveThread();
        }
        // CPython lives for the process lifetime. Finalizing here would break
        // other CostModel instances or an interpreter owned by the host.
    });
}
struct Gil {
    PyGILState_STATE state;
    Gil():state(PyGILState_Ensure()){}
    ~Gil(){PyGILState_Release(state);}
};
[[noreturn]] void python_error(const std::string& context) {
    PyObject *type=nullptr,*value=nullptr,*traceback=nullptr;
    PyErr_Fetch(&type,&value,&traceback);
    PyErr_NormalizeException(&type,&value,&traceback);
    Ref type_ref(type),value_ref(value),traceback_ref(traceback);
    Ref text(value?PyObject_Str(value):nullptr);
    const char* message=text?PyUnicode_AsUTF8(text.get()):nullptr;
    std::string detail=message?message:"unknown Python error";
    PyErr_Clear();
    throw std::runtime_error(context+": "+detail);
}
Ref checked(PyObject* p,const std::string& context) {
    if(!p)python_error(context);
    return Ref(p);
}
std::string text(PyObject* p) {
    const char* s=PyUnicode_AsUTF8(p);
    if(!s)python_error("reading Python string");
    return s;
}
int int_field(PyObject* dict,const char* key) {
    PyObject* item=PyDict_GetItemString(dict,key);
    if(!item)throw std::runtime_error(std::string("missing model field: ")+key);
    long value=PyLong_AsLong(item);
    if(PyErr_Occurred())python_error(std::string("reading model field ")+key);
    return static_cast<int>(value);
}
}
struct CostModel::Impl {
    PyObject *model=nullptr,*loads=nullptr,*dumps=nullptr,*generate=nullptr;
    std::mutex mutex;
    std::unordered_map<std::string,Prediction> cache;
    explicit Impl(const std::string& root) {
        initialize_python();
        Gil gil;
        auto path=checked(PyUnicode_FromString(root.c_str()),"encoding package path");
        PyObject* sys_path=PySys_GetObject("path");
        if(!sys_path||PyList_Insert(sys_path,0,path.get())<0)python_error("adding cost_model to sys.path");
        auto api=checked(PyImport_ImportModule("matmul_cost_model"),"importing matmul_cost_model");
        auto klass=checked(PyObject_GetAttrString(api.get(),"MatmulCostModel"),"loading MatmulCostModel");
        auto instance=checked(PyObject_CallObject(klass.get(),nullptr),"initializing calibrated model");
        auto generator=checked(PyObject_GetAttrString(api.get(),"generate_candidates"),"loading candidate generator");
        auto json=checked(PyImport_ImportModule("json"),"importing json");
        auto parser=checked(PyObject_GetAttrString(json.get(),"loads"),"loading json.loads");
        auto serializer=checked(PyObject_GetAttrString(json.get(),"dumps"),"loading json.dumps");
        model=instance.release();generate=generator.release();loads=parser.release();dumps=serializer.release();
    }
    ~Impl() {
        Gil gil;
        Py_XDECREF(model);Py_XDECREF(loads);Py_XDECREF(dumps);Py_XDECREF(generate);
    }
    Prediction evaluate(PyObject* input,bool semantic) {
        auto serialized=checked(PyObject_CallFunctionObjArgs(dumps,input,nullptr),"serializing config");
        std::string key=std::string(semantic?"semantic:":"calibrated:")+text(serialized.get());
        auto it=cache.find(key);
        if(it!=cache.end())return it->second;
        auto method=checked(PyObject_GetAttrString(model,"predict"),"loading predict");
        auto args=checked(PyTuple_Pack(1,input),"building prediction arguments");
        auto kwargs=checked(Py_BuildValue("{s:O}","semantic",semantic?Py_True:Py_False),"building prediction options");
        auto result=checked(PyObject_Call(method.get(),args.get(),kwargs.get()),"predicting matmul runtime");
        PyObject* value=PyDict_GetItemString(result.get(),"predicted_us");
        if(!value)throw std::runtime_error("Python prediction has no predicted_us");
        double us=PyFloat_AsDouble(value);
        if(PyErr_Occurred())python_error("reading predicted_us");
        if(!std::isfinite(us)||us<=0)throw std::runtime_error("invalid calibrated prediction");
        int active=int_field(result.get(),"active_compute_cores");
        auto detail=checked(PyObject_CallFunctionObjArgs(dumps,result.get(),nullptr),"serializing prediction");
        Prediction prediction{us,active,text(detail.get())};
        if(cache.size()>=8192)cache.clear();
        cache.emplace(std::move(key),prediction);
        return prediction;
    }
};
CostModel::CostModel(const std::string& root):impl_(std::make_unique<Impl>(root)){}
CostModel::~CostModel()=default;
CostModel::CostModel(CostModel&&) noexcept=default;
CostModel& CostModel::operator=(CostModel&&) noexcept=default;
Prediction CostModel::predict(const Config& config,bool semantic) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Gil gil;
    auto input=checked(Py_BuildValue("{s:s,s:[iii],s:s,s:i}",
        "implementation",config.implementation.c_str(),"shape",config.shape[0],config.shape[1],config.shape[2],
        "dtype",config.dtype.c_str(),"cores",config.cores),"building config");
    Ref tiling;
    if(config.implementation=="catlass")tiling=checked(Py_BuildValue("{s:s}","preset",config.catlass_preset.c_str()),"building CATLASS tiling");
    else tiling=checked(Py_BuildValue("{s:i,s:i,s:i,s:i}","base_m",config.tiling.base_m,"base_n",config.tiling.base_n,"base_k",config.tiling.base_k,"step_k",config.tiling.step_k),"building native tiling");
    if(PyDict_SetItemString(input.get(),"tiling",tiling.get())<0)python_error("setting tiling");
    return impl_->evaluate(input.get(),semantic);
}
Prediction CostModel::predict_json(const std::string& json,bool semantic) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Gil gil;
    auto encoded=checked(PyUnicode_FromString(json.c_str()),"encoding input JSON");
    auto input=checked(PyObject_CallFunctionObjArgs(impl_->loads,encoded.get(),nullptr),"parsing input JSON");
    return impl_->evaluate(input.get(),semantic);
}
std::vector<Config> CostModel::candidates(const Config& workload) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Gil gil;
    auto shape=checked(Py_BuildValue("[iii]",workload.shape[0],workload.shape[1],workload.shape[2]),"building shape");
    auto args=checked(PyTuple_Pack(1,shape.get()),"building candidate arguments");
    auto kwargs=checked(Py_BuildValue("{s:s,s:(s),s:(i)}","dtype",workload.dtype.c_str(),"implementations",workload.implementation.c_str(),"cores",workload.cores),"building candidate options");
    auto rows=checked(PyObject_Call(impl_->generate,args.get(),kwargs.get()),"generating reviewed candidates");
    std::vector<Config> result;
    for(Py_ssize_t i=0;i<PyList_Size(rows.get());++i) {
        PyObject* row=PyList_GetItem(rows.get(),i);
        Config c=workload;
        c.implementation=text(PyDict_GetItemString(row,"implementation"));
        if(c.implementation=="catlass")c.catlass_preset=text(PyDict_GetItemString(row,"tile"));
        else {
            PyObject* plan=PyDict_GetItemString(row,"plan");
            PyObject* tiling=PyDict_GetItemString(plan,"tcube");
            c.tiling={int_field(tiling,"baseM"),int_field(tiling,"baseN"),int_field(tiling,"baseK"),int_field(tiling,"stepKa")};
        }
        result.push_back(c);
    }
    return result;
}
}
