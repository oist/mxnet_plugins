#include "multi_regression-inl.h"
#include "mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(MultiRegressionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.regr_type) {
      case mreg_enum::kLinear:
        return new MultiRegressionOp<gpu, mshadow::op::identity, mshadow::op::minus, DType>();
      case mreg_enum::kLogistic:
        return new MultiRegressionOp<gpu, mshadow_op::sigmoid, mshadow::op::minus, DType>();
      case mreg_enum::kAngle:
        return new MultiRegressionOp<gpu, mod_pi, sin_minus, DType>();
      default:
        LOG(FATAL) << "unknown activation type " << param.regr_type;
    }
  })
  return op;
}
}  // namespace op
}  // namespace mxnet
