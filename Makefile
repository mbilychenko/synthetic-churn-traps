.PHONY: data verify analysis test all
data:      ## rebuild data/ from config/simulation.toml
	python run_pipeline.py --regenerate
analysis:  ## verify checksums and run all analyses
	python run_pipeline.py
test:
	pytest
all: data test
