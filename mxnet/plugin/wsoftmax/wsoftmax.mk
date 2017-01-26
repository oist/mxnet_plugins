WSOFTMAX_SRC = $(wildcard plugin/wsoftmax/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(WSOFTMAX_SRC))
WSOFTMAX_CUSRC = $(wildcard plugin/wsoftmax/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(WSOFTMAX_CUSRC))
