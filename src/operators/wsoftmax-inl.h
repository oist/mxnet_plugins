#ifndef PLUGIN_WSOFTMAX_WSOFTMAX_INL_H_
#define PLUGIN_WSOFTMAX_WSOFTMAX_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../../src/operator/operator_common.h"

namespace mxnet {
namespace op {

namespace wsoftmax_enum {
  enum WSoftmaxOpInputs {kData, kLabel}; //, kWeight};
  enum WSoftmaxOpOutputs {kOut};
  enum WSoftmaxOpResource {kTempSpace};
}  // namespace wsoftmax_enum

struct WSoftmaxParam : public dmlc::Parameter<WSoftmaxParam> {
  bool multi_output;
  DMLC_DECLARE_PARAMETER(WSoftmaxParam) {
    DMLC_DECLARE_FIELD(multi_output).set_default(true)
      .describe("If set to true, for a (n,k,x_1,..,x_n) dimensional "
        "input tensor, softmax will generate n*x_1*...*x_n output, each "
        "has k classes");
    };
  };

template<typename xpu, typename DType>
  class WSoftmaxOp : public Operator {
   public:
    explicit WSoftmaxOp(WSoftmaxParam param) : param_(param) {}

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(in_data.size(), 2) << "WSoftmax Input: [data, label]";
      CHECK_EQ(out_data.size(), 1) << "WSoftmax Output: [output]";
      Stream<xpu> *s = ctx.get_stream<xpu>();
      if (param_.multi_output) {
        int n = in_data[wsoftmax_enum::kData].size(0);
        int k = in_data[wsoftmax_enum::kData].size(1);
        Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[wsoftmax_enum::kData].Size()/n/k));
        Tensor<xpu, 3, DType> data =
            in_data[wsoftmax_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
        Tensor<xpu, 3, DType> out =
            out_data[wsoftmax_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
        Softmax(out, data);
      } else {
        Tensor<xpu, 2, DType> data = in_data[wsoftmax_enum::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[wsoftmax_enum::kOut].FlatTo2D<xpu, DType>(s);
        Softmax(out, data);
      }
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(in_data.size(), 2);
      CHECK_EQ(out_grad.size(), 1);
      CHECK_GE(in_grad.size(), 1);
      CHECK_GE(req.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();
      if (param_.multi_output) {
        int n = out_data[wsoftmax_enum::kOut].size(0);
        int k = out_data[wsoftmax_enum::kOut].size(1);
        Shape<3> s3 = Shape3(n, k, static_cast<int>(out_data[wsoftmax_enum::kOut].Size()/n/k));
        Shape<2> s2 = Shape2(n, static_cast<int>(out_data[wsoftmax_enum::kOut].Size()/n/k));
        Shape<3> sl = Shape3(n, 2, static_cast<int>(out_data[wsoftmax_enum::kOut].Size()/n/k));
        Tensor<xpu, 3, DType> label_weight = in_data[wsoftmax_enum::kLabel].get_with_shape<xpu, 3, DType>(sl, s);
        Tensor<xpu, 3, DType> out = out_data[wsoftmax_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
        Tensor<xpu, 3, DType> grad = in_grad[wsoftmax_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);

        sl[1] = 1;
        Tensor<xpu, 3, DType> label3 = ctx.requested[wsoftmax_enum::kTempSpace].get_space_typed<xpu, 3, DType>(sl, s);
        label3 = slice<1>(label_weight, 0, 1);
        Tensor<xpu, 2, DType> label = ctx.requested[wsoftmax_enum::kTempSpace].get_space_typed<xpu, 2, DType>(s2, s);
        label = reshape(label3, s2);

        SoftmaxGrad(grad, out, label);

        for (int i = 0; i < k; i++)
          slice<1>(grad, i, i+1) *= slice<1>(label_weight, 1, 2);

      } else {
        int n = out_data[wsoftmax_enum::kOut].size(0);
        int k = out_data[wsoftmax_enum::kOut].size(1);
        Shape<2> sl = Shape2(n, 2);
        Tensor<xpu, 2, DType> label_weight = in_data[wsoftmax_enum::kLabel].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[wsoftmax_enum::kOut].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> grad = in_grad[wsoftmax_enum::kData].FlatTo2D<xpu, DType>(s);

        sl[1] = 1;
        Tensor<xpu, 2, DType> label2 = ctx.requested[wsoftmax_enum::kTempSpace].get_space_typed<xpu, 2, DType>(sl, s);
        label2 = slice<1>(label_weight, 0, 1);
        Shape<1> s1 = Shape1(sl[0]);
        Tensor<xpu, 1, DType> label = ctx.requested[wsoftmax_enum::kTempSpace].get_space_typed<xpu, 1, DType>(s1, s);
        label = reshape(label2, s1);

        SoftmaxGrad(grad, out, label);

        for (int i = 0; i < k; i++)
          slice<1>(grad, i, i+1) *= slice<1>(label_weight, 1, 2);
      }
  }

  private:
    WSoftmaxParam param_;
};

template<typename xpu> Operator *CreateOp(WSoftmaxParam param, int dtype);

#if DMLC_USE_CXX11

class WSoftmaxProp : public OperatorProperty {
  public:

    std::vector<std::string> ListArguments() const override {
      return {"data", "label"};
    }

    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
       param_.Init(kwargs);
    }

    std::map<std::string, std::string> GetParams() const override {
      return param_.__DICT__();
    }

    bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
      using namespace mshadow;
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
      const TShape &dshape = (*in_shape)[wsoftmax_enum::kData];
      if (dshape.ndim() == 0) return false;
      if (param_.multi_output) {
        SHAPE_ASSIGN_CHECK(*in_shape, wsoftmax_enum::kLabel,
                           Shape3(dshape[0], 2, dshape.Size()/dshape[0]/dshape[1]));
      } else {
        TShape label_shape(dshape.ndim());
        for (index_t i = 0; i + 1 < dshape.ndim(); ++i)
          label_shape[i] = dshape[i];
        label_shape[dshape.ndim() - 1] = 2;
        SHAPE_ASSIGN_CHECK(*in_shape, wsoftmax_enum::kLabel, label_shape);
      }
      out_shape->clear();
      out_shape->push_back(dshape);
      return true;
    }

    bool InferType(std::vector<int> *in_type,
                    std::vector<int> *out_type,
                    std::vector<int> *aux_type) const override {
       CHECK_GE(in_type->size(), 1);
       int dtype = (*in_type)[0];
       CHECK_NE(dtype, -1) << "First input must have specified type";
       for (index_t i = 0; i < in_type->size(); ++i) {
         if ((*in_type)[i] == -1) {
           (*in_type)[i] = dtype;
         } else {
           CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                          << "Expected " << dtype << " v.s. given "
                                          << (*in_type)[i] << " at " << ListArguments()[i];
         }
       }
       out_type->clear();
       out_type->push_back(dtype);
       return true;
    }

    OperatorProperty* Copy() const override {
      auto ptr = new WSoftmaxProp();
      ptr->param_ = param_;
      return ptr;
    }

    std::string TypeString() const override {
      return "WSoftmax";
    }

    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {in_data[wsoftmax_enum::kLabel], out_data[wsoftmax_enum::kOut]};
    }

    std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const override {
      return {{out_data[wsoftmax_enum::kOut], in_grad[wsoftmax_enum::kData]}};
    }

    std::vector<std::pair<int, void*> > ForwardInplaceOption(
       const std::vector<int> &in_data,
       const std::vector<void*> &out_data) const override {
       return {{in_data[wsoftmax_enum::kData], out_data[wsoftmax_enum::kOut]}};
     }

    std::vector<ResourceRequest> BackwardResource(
         const std::vector<TShape> &in_shape) const override {
      return {ResourceRequest::kTempSpace};
    }

    Operator* CreateOperator(Context ctx) const override {
       LOG(FATAL) << "Not Implemented.";
       return NULL;
    }

    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                std::vector<int> *in_type) const override;

 protected:
   WSoftmaxParam param_;
};  // class WSoftmaxProp

#endif  // DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // PLUGIN_WSOFTMAX_WSOFTMAX_INL_H_
