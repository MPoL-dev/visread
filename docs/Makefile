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

NOTEBOOKS := tutorials/introduction_to_casatools.ipynb tutorials/plot_baselines.ipynb tutorials/AS209.ipynb 

clean:
	rm -rf _build
	rm -rf ${NOTEBOOKS}
	rm -rf tutorials/.ipynb_checkpoints
	rm -rf tutorials/runs

tutorials/%.ipynb: tutorials/%.py 
	jupytext --to ipynb --execute $<

html: ${NOTEBOOKS}
	python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
