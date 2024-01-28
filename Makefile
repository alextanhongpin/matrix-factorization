vw := poetry run python -m vowpalwabbit

jupyter:
	@poetry run jupyter-lab


convert_all:
	# jupytext doesn't preserve image.
	#@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupytext --to md {} \;
	@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupyter nbconvert --to markdown --output-dir=docs {} \;

# Similar to convert, but only convert the diff files.
convert:
	@poetry run jupyter nbconvert --to markdown --output-dir=docs $(shell git diff HEAD --name-only | grep .ipynb)
