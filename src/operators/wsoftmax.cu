#include "./wsoftmax-inl.h"
#include <stdio.h>
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(WSoftmaxParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new WSoftmaxOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet
