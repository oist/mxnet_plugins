#include "wsoftmax-inl.h"
#include <vector>

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(WSoftmaxParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new WSoftmaxOp<cpu, DType>(param);
  })
  return op;
}

Operator *WSoftmaxProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(WSoftmaxParam);

MXNET_REGISTER_OP_PROPERTY(WSoftmax, WSoftmaxProp)
.describe("Perform a softmax transformation on input, backprop with logloss weighted by class-dependent weight map.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_argument("label", "Symbol", "Label data, can also be "\
              "probability value with same shape as data")
.add_arguments(WSoftmaxParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
