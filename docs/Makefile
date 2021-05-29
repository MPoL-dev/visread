# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html clean

NOTEBOOKS := tutorials/introduction_to_casatools.ipynb tutorials/rescale_AS209_weights.ipynb 

tutorials/AS209_MS/AS209.fits: tutorials/dl_and_tclean_AS209.py
	python tutorials/dl_and_tclean_AS209.py

clean:
	rm -rf _build
	rm -rf ${NOTEBOOKS}
	rm -rf tutorials/.ipynb_checkpoints
	rm -rf tutorials/runs

tutorials/%.ipynb: tutorials/%.py tutorials/AS209_MS/AS209.fits
	jupytext --to ipynb --execute $<

html: ${NOTEBOOKS}
	python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
