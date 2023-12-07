# Minimal makefile for Sphinx documentation

tutorials/AS209_MS/AS209.fits: tutorials/dl_and_tclean_AS209.py
	python tutorials/dl_and_tclean_AS209.py

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

clean:
	rm -rf _build
	rm -rf casa*log
	rm -rf tutorials/AS209_MS

html: tutorials/AS209_MS/AS209.fits
	sphinx-build -T -b html -d _build/doctrees -D language=en . _build/html