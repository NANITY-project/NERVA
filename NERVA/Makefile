CXX      ?= g++
# NATIVE=1 (default) tunes for this CPU (AVX2/FMA etc). Use NATIVE=0 for a portable binary.
NATIVE   ?= 1
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fno-math-errno -fno-trapping-math -Iinclude
ifeq ($(NATIVE),1)
CXXFLAGS += -march=native
endif
HDRS := $(wildcard include/*.hpp)

all: ik_demo ik_bench ik_quality ik_body_test lib

# Shared library for host apps (portable: no -march=native).
lib: build/libnebula.so
build/libnebula.so: capi/nebula_capi.cpp capi/nebula_capi.h $(HDRS)
	@mkdir -p build
	$(CXX) -std=c++17 -O3 -fPIC -shared -fvisibility=hidden -fno-math-errno -fno-trapping-math -Iinclude -Icapi capi/nebula_capi.cpp -o $@

ik_demo: src/main.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) src/main.cpp -o $@

ik_bench: src/bench.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) src/bench.cpp -o $@

ik_quality: src/quality.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) src/quality.cpp -o $@

ik_body_test: src/body_test.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) src/body_test.cpp -o $@

body-test: ik_body_test
	./ik_body_test

test: ik_demo
	./ik_demo

bench: ik_bench
	./ik_bench 1024
	./ik_bench 1024 idle

# Address + undefined-behaviour sanitizer run of the test suite.
sanitize: src/main.cpp $(HDRS)
	$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -Iinclude src/main.cpp -o ik_demo_san
	./ik_demo_san
	$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -Iinclude src/body_test.cpp -o ik_body_san
	./ik_body_san

clean:
	rm -rf ik_demo ik_bench ik_quality ik_body_test ik_demo_san ik_body_san build

# --- Studio + Blender -------------------------------------------------------
studio: lib
	python3 studio/server.py

# Real-NEON tests need a NEON binary (auto-detected next to this project, or set NEON_BIN / NEON_SRC); they skip without one.
NEON_BIN_FOUND := $(shell python3 -c "import sys; sys.path.insert(0,'studio'); import neon_bridge as n; print(n.find_binary() or '')" 2>/dev/null)

studio-test: lib
	python3 studio/tests/test_sim.py
	python3 studio/tests/test_agent.py
	python3 studio/tests/test_gltf.py
	NEON_BIN="$${NEON_BIN:-$(NEON_BIN_FOUND)}" python3 studio/tests/test_neon.py
	NEON_BIN="$${NEON_BIN:-$(NEON_BIN_FOUND)}" python3 studio/tests/test_server.py
	cd studio/tests/ui && npm install --no-audit --no-fund --silent && node --import ./register.mjs test_logic_viewport.mjs && node --import ./register.mjs test_note.mjs && node --import ./register.mjs test_avatar.mjs && NEON_BIN="$${NEON_BIN:-$(NEON_BIN_FOUND)}" node --import ./register.mjs test_ui.mjs

blender-test: lib
	./blender/build_addon.sh
	blender -b --factory-startup --python blender/test_addon.py

test-all: test body-test studio-test blender-test

.PHONY: all test bench sanitize clean lib studio studio-test blender-test test-all body-test
