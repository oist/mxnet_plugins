#ifndef MXNET_OPERATOR_MULTI_REGRESSION_INL_H_
#define MXNET_OPERATOR_MULTI_REGRESSION_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <math.h>
#include "../../src/operator/operator_common.h"

namespace mxnet {
namespace op {


struct mod_pi {
  template<typename DType>
    MSHADOW_XINLINE static DType Map(DType a) {
      if (a < 0.0f) return a;
      return DType(fmod(a, DType(2*M_PI)));
  }
};


struct sin_minus {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    //if (a < 0.0f && b < 0.0f) return a-b;
    if (a >= 0.0f && b >= 0.0f) return DType(sinf((a - b)/DType(2.0f)));
    return a-b; //DType(2.0f);
  }
};

namespace mreg_enum {
enum MultiRegressionOpInputs {kData, kLabel};
enum MultiRegressionOutputs {kOut};
enum MultiRegressionType {kLinear, kLogistic, kAngle};
//enum MultiRegressionOpResource {kTempSpace};
}  // mreg_enum

struct MultiRegressionParam : public dmlc::Parameter<MultiRegressionParam> {
  int regr_type;
  DMLC_DECLARE_PARAMETER(MultiRegressionParam) {
    DMLC_DECLARE_FIELD(regr_type)
    .add_enum("linear", mreg_enum::kLinear)
    .add_enum("logistic", mreg_enum::kLogistic)
    .add_enum("angle", mreg_enum::kAngle)
    .describe("Type of regression.");
  }
};

// Special Operator to output regression value in forward
// And get gradient in calculation.
template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
class MultiRegressionOp : public Operator {
 public:
  //explicit MultiRegressionOp(MultiRegressionParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "MultiRegressionOp Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "MultiRegressionOp Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    //Tensor<xpu, 2> data = in_data[mreg_enum::kData].FlatTo2D<xpu, real_t>(s);
    //Tensor<xpu, 2> out = out_data[mreg_enum::kOut].FlatTo2D<xpu, real_t>(s);

    int n = in_data[mreg_enum::kData].size(0);
    //int k = in_data[mreg_enum::kData].size(1);
    Shape<3> s3 = Shape3(n, 1, static_cast<int>(in_data[mreg_enum::kData].Size()/n));
    Tensor<xpu, 3, DType> data = in_data[mreg_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out = out_data[mreg_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);

    Assign(out, req[mreg_enum::kOut], F<ForwardOp>(data));
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

    //? real_t num_output = in_data[mreg_enum::kLabel].Size()/in_data[mreg_enum::kLabel].shape_[0];

    int n = out_data[mreg_enum::kOut].size(0);
    //int k = out_data[mreg_enum::kOut].size(1);
    Shape<3> s3 = Shape3(n, 1, static_cast<int>(out_data[mreg_enum::kOut].Size()/n));
    //Shape<2> s2 = Shape2(n, static_cast<int>(out_data[mreg_enum::kOut].Size()/n/k));
    Shape<3> sl = Shape3(n, 2, static_cast<int>(out_data[mreg_enum::kOut].Size()/n));
    Tensor<xpu, 3, DType> label_weight = in_data[mreg_enum::kLabel].get_with_shape<xpu, 3, DType>(sl, s);
    Tensor<xpu, 3, DType> out = out_data[mreg_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> grad = in_grad[mreg_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);

    //sl[1] = 1;
    Assign(grad, req[mreg_enum::kData],
           slice<1>(label_weight, 1, 2) * F<BackwardOp>(out, slice<1>(label_weight, 0, 1)));

//    for (int i = 0; i < k; i++)
//      Assign(slice<1>(grad, i, i+1), req[mreg_enum::kData],
//             slice<1>(label_weight, 1, 2) * F<BackwardOp>(slice<1>(out, i, i+1), slice<1>(label_weight, 0, 1))); //reshape(label3, s2))); //reshape(label, grad.shape_)));

    //for (int i = 0; i < k; i++)
    //  slice<1>(grad, i, i+1) *= slice<1>(label_weight, 1, 2);
  }

};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(MultiRegressionParam param, int dtype);

#if DMLC_USE_CXX11
class MultiRegressionProp : public OperatorProperty {
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
    const TShape &dshape = (*in_shape)[mreg_enum::kData];

    SHAPE_ASSIGN_CHECK(*in_shape, mreg_enum::kLabel,
                       Shape3(dshape[0], 2, dshape.Size()/dshape[0]));

    TShape oshape = dshape;
    oshape[1] = 1;
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiRegressionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MultiRegression";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[mreg_enum::kLabel], out_data[mreg_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[mreg_enum::kOut], in_grad[mreg_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[mreg_enum::kData], out_data[mreg_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  MultiRegressionParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MULTI_REGRESSION_INL_H_
