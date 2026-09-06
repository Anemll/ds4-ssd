CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal -framework IOSurface
CORE_OBJS = ds4.o ds4_profile.o ds4_metal.o ds4_ane_mlp_int8w.o
CPU_CORE_OBJS = ds4_cpu.o ds4_profile.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CORE_OBJS = ds4.o ds4_profile.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_profile.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression ane-smoke sidecar-smoke flash-moe-slot-test flash-moe-slot-test-sanitize flash-moe-io-test

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make ane-smoke    Build and run the ANE int8 MLP precision smoke"
	@echo "  make sidecar-smoke Run the 4K SSD sidecar smoke (requires DS4_SIDECAR_DIR)"
	@echo "  make test         Build and run tests"
	@echo "  make flash-moe-slot-test Run model-free Flash-MoE slot regressions"
	@echo "  make flash-moe-io-test Run model-free Flash-MoE I/O lifecycle regressions"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_ane_mlp_int8w.o: ds4_ane_mlp_int8w.m ds4_ane_mlp_int8w.h
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_ane_mlp_int8w.m

tests/ane_ds4_mlp_i8i8_precision_smoke: tests/ane_ds4_mlp_i8i8_precision_smoke.m ds4_ane_mlp_int8w.o
	$(CC) -fobjc-arc -O2 -I. -o $@ tests/ane_ds4_mlp_i8i8_precision_smoke.m ds4_ane_mlp_int8w.o -framework Foundation -framework IOSurface -lpthread

ane-smoke: tests/ane_ds4_mlp_i8i8_precision_smoke
	./tests/ane_ds4_mlp_i8i8_precision_smoke

tests/mxfp4_native_probe: tests/mxfp4_native_probe.m metal/mxfp4_common.h metal/mxfp4_native.metal metal/moe.metal
	$(CC) -fobjc-arc -O2 -I. -o $@ tests/mxfp4_native_probe.m -framework Foundation -framework Metal -framework QuartzCore

mxfp4-native-probe: tests/mxfp4_native_probe
	./tests/mxfp4_native_probe

sidecar-smoke: ds4
	DS4_METAL_PREFILL_CHUNK=4096 ./tests/sidecar_smoke.sh

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make ane-smoke           Requires macOS private ANE framework"
	@echo "  make sidecar-smoke       Requires macOS Metal"
	@echo "  make test                Build and run tests"
	@echo "  make flash-moe-slot-test Run model-free Flash-MoE slot regressions"
	@echo "  make clean               Remove build outputs"

ane-smoke:
	@echo "ane-smoke requires macOS private ANE framework"
	@exit 2

sidecar-smoke:
	@echo "sidecar-smoke requires macOS Metal"
	@exit 2

cuda-spark:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

DS4_INCLUDED_SRCS = \
	dspark.c \
	dspark.h \
	ds4_metal_diagnostics.c \
	ds4_metal_mtp.c \
	glm52/glm52_model.c \
	glm52/glm52_runtime.c \
	hy3/hy3_model.c \
	hy3/hy3_runtime.c \
	hy4/hy4_model.c \
	hy4/hy4_runtime.c \
	hy4/hy4_math.h \
	ssd/ssd_flash_moe_sidecar.c \
	ssd/ssd_flash_moe_streaming.c \
	ssd/ssd_flash_moe_allocation.c \
	ssd/ssd_flash_moe_runtime.c \
	ssd/ssd_flash_moe_resident_prefill.c \
	ssd/ssd_flash_moe_slot_cache.c \
	ssd/ssd_flash_moe_slots.h \
	ssd/ssd_flash_moe_decode.c \
	ssd/ssd_flash_moe_prefill.c \
	ssd/ssd_flash_moe_diagnostics.c \
	ssd/ssd_flash_moe_mxfp4_slots6.c

ds4.o: ds4.c $(DS4_INCLUDED_SRCS) ds4.h ds4_gpu.h ds4_profile.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_profile.o: ds4_profile.c ds4_profile.h
	$(CC) $(CFLAGS) -c -o $@ ds4_profile.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_agent.o: ds4_agent.c ds4.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c $(DS4_INCLUDED_SRCS) ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

tests/test_flash_moe_slots: tests/test_flash_moe_slots.c ssd/ssd_flash_moe_slots.h
	$(CC) $(CFLAGS) -o $@ $<

tests/test_flash_moe_slots_sanitize: tests/test_flash_moe_slots.c ssd/ssd_flash_moe_slots.h
	$(CC) -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ $<

flash-moe-slot-test: tests/test_flash_moe_slots
	./tests/test_flash_moe_slots

flash-moe-slot-test-sanitize: tests/test_flash_moe_slots_sanitize
	./tests/test_flash_moe_slots_sanitize

ifeq ($(UNAME_S),Darwin)
tests/test_flash_moe_io.o: tests/test_flash_moe_io.c ds4.c $(DS4_INCLUDED_SRCS) ds4.h ds4_gpu.h ds4_profile.h
	$(CC) $(CFLAGS) -O1 -UNDEBUG -Wno-unused-function -Wno-unused-parameter -c -o $@ $<

tests/test_flash_moe_io: tests/test_flash_moe_io.o ds4_profile.o ds4_metal.o ds4_ane_mlp_int8w.o
	$(CC) -o $@ $^ $(METAL_LDLIBS)

flash-moe-io-test: tests/test_flash_moe_io
	./tests/test_flash_moe_io

# Explicit opt-in: callers choose a local DS4 sidecar and I/O settings.
tests/test_flash_moe_session: tests/test_flash_moe_session.c $(CORE_OBJS) ds4.h
	$(CC) $(CFLAGS) -o $@ $< $(CORE_OBJS) $(METAL_LDLIBS)

.PHONY: flash-moe-session-test
flash-moe-session-test: tests/test_flash_moe_session
	@test -n "$(DS4_SIDECAR_DIR)" || (echo "set DS4_SIDECAR_DIR to a local DS4 package"; exit 2)
	./tests/test_flash_moe_session "$(DS4_SIDECAR_DIR)"

test: flash-moe-io-test
else
flash-moe-io-test:
	@echo "flash-moe-io-test requires macOS Metal support objects"
	@exit 2
endif

test: ds4_test flash-moe-slot-test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/ane_ds4_mlp_i8i8_precision_smoke tests/test_flash_moe_slots tests/test_flash_moe_slots_sanitize tests/test_flash_moe_io tests/test_flash_moe_io.o tests/test_flash_moe_session

.PHONY: hy4-math-test hy4-quant-test
tests/test_hy4_math: tests/test_hy4_math.c hy4/hy4_math.h
	$(CC) $(CFLAGS) -o $@ $< -lm
hy4-math-test: tests/test_hy4_math
	./tests/test_hy4_math

tests/test_hy4_quants: tests/test_hy4_quants.c hy4/hy4_quants.c hy4/hy4_quants.h hy4/hy4_quant_tables.h ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o
	$(CC) -O2 -Wall -Wextra -std=c99 -o $@ tests/test_hy4_quants.c hy4/hy4_quants.c ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o $(METAL_LDLIBS)
hy4-quant-test: tests/test_hy4_quants
	./tests/test_hy4_quants --metal

tests/test_hy4_session: tests/test_hy4_session.c $(CORE_OBJS) ds4.h
	$(CC) $(CFLAGS) -o $@ $< $(CORE_OBJS) $(METAL_LDLIBS)

tests/test_hy4_metadata: tests/test_hy4_metadata.c ds4.c $(DS4_INCLUDED_SRCS) ds4.h ds4_profile.o ds4_metal.o ds4_ane_mlp_int8w.o
	$(CC) -O2 $(NATIVE_CPU_FLAG) -std=c99 -ffunction-sections -fdata-sections -Wno-unused-function -Wno-unused-parameter -Wl,-dead_strip -o $@ $< ds4_profile.o ds4_metal.o ds4_ane_mlp_int8w.o $(METAL_LDLIBS)
.PHONY: hy4-metadata-test
hy4-metadata-test: tests/test_hy4_metadata
	@test -n "$(HY4_MODEL)" || (echo "set HY4_MODEL to the HY4 dense GGUF"; exit 2)
	./tests/test_hy4_metadata "$(HY4_MODEL)"

tests/test_hy4_attention: tests/test_hy4_attention.c hy4/hy4_math.h ds4_gpu.h ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o
	$(CC) -O2 -Wall -Wextra -std=c99 -o $@ $< ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o $(METAL_LDLIBS)
.PHONY: hy4-attention-test
hy4-attention-test: tests/test_hy4_attention
	./tests/test_hy4_attention

tests/test_hy4_math_sanitize: tests/test_hy4_math.c hy4/hy4_math.h
	$(CC) -O1 -g -std=c99 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ $< -lm
tests/test_hy4_quants_sanitize: tests/test_hy4_quants.c hy4/hy4_quants.c hy4/hy4_quants.h hy4/hy4_quant_tables.h
	$(CC) -O1 -g -std=c99 -Wall -Wextra -DHY4_TEST_CPU_ONLY -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ tests/test_hy4_quants.c hy4/hy4_quants.c -lm
.PHONY: hy4-sanitize-test
hy4-sanitize-test: tests/test_hy4_math_sanitize tests/test_hy4_quants_sanitize
	./tests/test_hy4_math_sanitize
	./tests/test_hy4_quants_sanitize


tests/test_hy4_pointwise: tests/test_hy4_pointwise.c hy4/hy4_math.h ds4_gpu.h ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o
	$(CC) -O2 -Wall -Wextra -std=c99 -o $@ $< ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o $(METAL_LDLIBS)
.PHONY: hy4-pointwise-test
hy4-pointwise-test: tests/test_hy4_pointwise
	./tests/test_hy4_pointwise

# Native HY4 two-dispatch top-8 FFN; bounded synthetic banks, no model loading.
tests/test_hy4_fused: tests/test_hy4_fused.c hy4/hy4_quants.c hy4/hy4_quants.h ds4_gpu.h ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o
	$(CC) -O2 -Wall -Wextra -std=c99 -o $@ tests/test_hy4_fused.c hy4/hy4_quants.c ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o $(METAL_LDLIBS)
.PHONY: hy4-fused-test
hy4-fused-test: tests/test_hy4_fused
	./tests/test_hy4_fused

# Independent HY4 HC GPU math, no model loading.
tests/test_hy4_hc: tests/test_hy4_hc.c hy4/hy4_math.h ds4_gpu.h ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o
	$(CC) -O2 -Wall -Wextra -std=c99 -o $@ tests/test_hy4_hc.c ds4_metal.o ds4_profile.o ds4_ane_mlp_int8w.o $(METAL_LDLIBS)
.PHONY: hy4-hc-test
hy4-hc-test: tests/test_hy4_hc
	./tests/test_hy4_hc
