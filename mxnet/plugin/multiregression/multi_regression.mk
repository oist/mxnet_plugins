MULTIREGRESSION_SRC = $(wildcard plugin/multiregression/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(MULTIREGRESSION_SRC))
MULTIREGRESSION_CUSRC = $(wildcard plugin/multiregression/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(MULTIREGRESSION_CUSRC))
