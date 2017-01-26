#include "multi_regression-inl.h"
#include "mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(MultiRegressionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.regr_type) {
      case mreg_enum::kLinear:
        return new MultiRegressionOp<cpu, mshadow::op::identity, mshadow::op::minus, DType>();
      case mreg_enum::kLogistic:
        return new MultiRegressionOp<cpu, mshadow_op::sigmoid, mshadow::op::minus, DType>();
      case mreg_enum::kAngle:
        return new MultiRegressionOp<cpu, mod_pi, sin_minus, DType>();
      default:
        LOG(FATAL) << "unknown retression type " << param.regr_type;
    }
  })
  return op;
}

Operator *MultiRegressionProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                      std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiRegressionParam);

MXNET_REGISTER_OP_PROPERTY(MultiRegression, MultiRegressionProp)
.describe("Use linear/logistic.angle regression for final output, this is used on final output of a net.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(MultiRegressionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
