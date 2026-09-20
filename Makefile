# AegisQ-HPC developer entry points.
#
#   make configure   configure the CMake build tree
#   make build       build the native core + Python extension
#   make test        C++ unit tests + single-process Python tests
#   make test-mpi    distributed correctness tests under mpirun
#   make benchmark   local benchmark sweep (writes benchmarks/raw)
#   make clean       remove build artefacts

PYTHON      ?= python3
BUILD_DIR   ?= build
BUILD_TYPE  ?= Release
MPI_RANKS   ?= 4
CMAKE_FLAGS ?=

.PHONY: all configure build install test test-cpp test-python test-mpi benchmark report paper lint format clean distclean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DPython_EXECUTABLE=$$( $(PYTHON) -c "import sys; print(sys.executable)" ) \
		$(CMAKE_FLAGS)

build: configure
	cmake --build $(BUILD_DIR) --parallel

install: build
	$(PYTHON) -m pip install -e .

test: test-cpp test-python

test-cpp: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-python:
	$(PYTHON) -m pytest tests -m "not slow and not mpi and not gpu"

test-mpi: build
	MPI_RANKS=$(MPI_RANKS) ./scripts/run_mpi_tests.sh

benchmark:
	./scripts/benchmark_local.sh

report:
	$(PYTHON) scripts/generate_report.py

paper: report
	@command -v tectonic >/dev/null && (cd paper && tectonic main.tex) || \
		echo "tectonic not installed; see paper/main.tex"

lint:
	$(PYTHON) -m ruff check aegisq tests scripts
	$(PYTHON) -m ruff format --check aegisq tests

format:
	$(PYTHON) -m ruff format aegisq tests
	@command -v clang-format >/dev/null && \
		clang-format -i cpp/src/*.cpp cpp/include/aegisq/*.hpp cpp/tests/*.cpp cpp/tests/*.hpp || \
		echo "clang-format not installed; skipped C++ formatting"

clean:
	rm -rf $(BUILD_DIR)
	find aegisq -name "_aegisq_core*.so" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf .pytest_cache .ruff_cache *.egg-info
