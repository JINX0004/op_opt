CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wpedantic -O2
CPPFLAGS ?= -Isrc
PYTHON_CONFIG ?= python3-config
CPPFLAGS += $(shell $(PYTHON_CONFIG) --includes)
CPPFLAGS += -DMATMUL_PYTHON_PATH=\"$(abspath ../cost_model)\"
LDLIBS += $(shell $(PYTHON_CONFIG) --embed --ldflags)
# Keep the selected interpreter's shared library discoverable at runtime.
comma := ,
LDFLAGS += $(patsubst -L%,-Wl$(comma)-rpath$(comma)%,$(filter -L%,$(shell $(PYTHON_CONFIG) --embed --ldflags)))

SRC_DIR   := src
BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin
TARGET    := $(BIN_DIR)/local_search

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run debug

all: $(TARGET)

debug:
	$(MAKE) BUILD_DIR=build/debug \
		CXXFLAGS="-std=c++17 -Wall -Wextra -Wpedantic -O0 -g"

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

-include $(DEPS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) local_search

.PHONY: test

test: $(TARGET) $(BIN_DIR)/test_genetic_algorithm $(BIN_DIR)/test_cost_model
	$(BIN_DIR)/test_genetic_algorithm
	$(BIN_DIR)/test_cost_model
	$(patsubst %-config,%,$(PYTHON_CONFIG)) tests/test_python_model.py $(abspath $(TARGET))

$(BIN_DIR)/test_genetic_algorithm: tests/test_genetic_algorithm.cpp src/generic_tiling.hpp | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/test_cost_model: tests/test_cost_model.cpp $(BUILD_DIR)/matmul_cost_model.o | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
