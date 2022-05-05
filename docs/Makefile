# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

test: Makefile
	@cd source && DOCTEST=1 python3 -c "import cn"

html: Makefile
	@rm -rf "$(BUILDDIR)-cn" && $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)-cn" $(SPHINXOPTS) $(O)

clean: Makefile
	@rm -rf build build-cn
