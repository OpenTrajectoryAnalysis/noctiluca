DOCDIR = doc
EXAMPLEDIR = examples
SPHINXDIR = $(DOCDIR)/sphinx
SPHINXSOURCE = $(SPHINXDIR)/source
SPHINXBUILD = $(SPHINXDIR)/source/_build
TESTDIR = tests
TESTFILE = test_noctiluca.py
COVERAGEREPFLAGS =
COVERAGEREPDIR = $(TESTDIR)/coverage
DISTDIR = dist
MODULE = noctiluca

.PHONY : build examples pre-docs docs tests all clean mydocs mytests myall myclean

all : docs tests

build :
	-@cd $(DISTDIR) && rm *
	python3 -m build

pre-docs :
	sphinx-apidoc -f -o $(SPHINXSOURCE) $(MODULE)
	@rm $(SPHINXSOURCE)/modules.rst
	@cd $(SPHINXSOURCE) && vim -nS post-apidoc.vim
	cd $(SPHINXDIR) && $(MAKE) clean
	-@rm $(SPHINXSOURCE)/$(EXAMPLEDIR)/*.ipynb
	@cp -rf $(EXAMPLEDIR) $(SPHINXSOURCE)/
	@cd $(SPHINXSOURCE) && vim -nS write_examples_rst.vim

docs : pre-docs
	cd $(SPHINXDIR) && $(MAKE) html

tests :
	cd $(TESTDIR) && coverage run $(TESTFILE)
	@mv $(TESTDIR)/.coverage .
	coverage html -d $(COVERAGEREPDIR) $(COVERAGEREPFLAGS)

clean :
	-rm -r $(SPHINXBUILD)/*
	-rm -r $(COVERAGEREPDIR)/*
	-rm .coverage

# Personal convenience targets
DUMPPATH = "/home/simongh/Dropbox (MIT)/htmldump"
mydocs : docs
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mytests : tests
	cp -r $(COVERAGEREPDIR)/* $(DUMPPATH)/coverage

myall : mydocs mytests

myclean : clean
	-rm -r $(DUMPPATH)/sphinx/*
	-rm -r $(DUMPPATH)/coverage/*
