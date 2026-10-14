# NERVA overlay (Wayland / Hyprland). Needs: g++, wayland-scanner, wayland-protocols, libwayland-dev.
#   make            builds build/nerva-overlay
#   make test       unit tests (no display needed)
CXX      ?= g++
CC       ?= cc
BUILD    := build
ROOT     := ..
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fno-math-errno -I$(ROOT)/include -Iinclude -I$(BUILD)/gen
PKGDATA  := $(shell pkg-config --variable=pkgdatadir wayland-protocols 2>/dev/null)
XDGXML   := $(PKGDATA)/stable/xdg-shell/xdg-shell.xml
WLRXML   := protocols/wlr-layer-shell-unstable-v1.xml
WLCFLAGS := $(shell pkg-config --cflags wayland-client 2>/dev/null)
WLLIBS   := $(shell pkg-config --libs wayland-client 2>/dev/null)
HDRS     := $(wildcard include/*.hpp) $(wildcard $(ROOT)/include/*.hpp)

all: $(BUILD)/nerva-overlay

$(BUILD)/gen/wlr-layer-shell-unstable-v1-client-protocol.h: $(WLRXML)
	@mkdir -p $(BUILD)/gen && wayland-scanner client-header $< $@
	@sed -i 's/\bnamespace\b/name_space/g' $@   # 'namespace' is a C++ keyword (an argument name in the generated header)
$(BUILD)/gen/wlr-layer-shell.c: $(WLRXML)
	@mkdir -p $(BUILD)/gen && wayland-scanner private-code $< $@
$(BUILD)/gen/xdg-shell.c: $(XDGXML)
	@mkdir -p $(BUILD)/gen && wayland-scanner private-code $< $@
$(BUILD)/wlr-layer-shell.o: $(BUILD)/gen/wlr-layer-shell.c
	$(CC) -O2 -c $< -o $@ $(WLCFLAGS)
$(BUILD)/xdg-shell.o: $(BUILD)/gen/xdg-shell.c
	$(CC) -O2 -c $< -o $@ $(WLCFLAGS)
$(BUILD)/nerva-overlay: src/overlay.cpp $(HDRS) $(BUILD)/gen/wlr-layer-shell-unstable-v1-client-protocol.h $(BUILD)/wlr-layer-shell.o $(BUILD)/xdg-shell.o
	$(CXX) $(CXXFLAGS) $(WLCFLAGS) src/overlay.cpp $(BUILD)/wlr-layer-shell.o $(BUILD)/xdg-shell.o -o $@ $(WLLIBS)

$(BUILD)/mouse_test: tests/mouse_test.cpp $(HDRS)
	@mkdir -p $(BUILD) && $(CXX) $(CXXFLAGS) -fsanitize=address,undefined -g tests/mouse_test.cpp -o $@

$(BUILD)/perception_test: tests/perception_test.cpp $(HDRS)
	@mkdir -p $(BUILD) && $(CXX) $(CXXFLAGS) -fsanitize=address,undefined -g tests/perception_test.cpp -o $@

$(BUILD)/overlay_test: tests/overlay_test.cpp $(HDRS)
	@mkdir -p $(BUILD) && $(CXX) $(CXXFLAGS) -fsanitize=address,undefined -g tests/overlay_test.cpp -o $@

test: $(BUILD)/overlay_test $(BUILD)/mouse_test $(BUILD)/perception_test $(BUILD)/nerva-overlay
	./$(BUILD)/overlay_test
	./$(BUILD)/mouse_test
	NERVA_TEST_DIR_OVERLAY=. ./$(BUILD)/perception_test
	python3 tests/test_actions.py
	python3 tests/test_a11y_helper.py
	python3 tests/test_integration.py
	python3 tests/test_a11y_full.py

clean:
	rm -rf $(BUILD)
.PHONY: all test clean
