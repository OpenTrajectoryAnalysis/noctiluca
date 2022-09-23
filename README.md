[![Documentation Status](https://readthedocs.org/projects/noctiluca/badge/?version=latest)](https://noctiluca.readthedocs.io/en/latest/?badge=latest)

noctiluca
=========

A python framework for trajectory analysis in single particle tracking (SPT)
experiments. For an
[Introduction](https://noctiluca.readthedocs.org/en/latest/intro.html), some
worked [examples](https://noctiluca.readthedocs.org/en/latest/examples.html),
and the full [API
reference](https://noctiluca.readthedocs.org/en/latest/noctiluca.html) head
over to our documentation at
[ReadTheDocs](https://noctiluca.readthedocs.org/en/latest).

To install `noctiluca` you can use the latest stable version from [PyPI](https://pypi.org/project/noctiluca)
```sh
$ pip install --upgrade noctiluca
```
or the very latest updates right from GitHub:
```sh
$ pip install git+https://github.com/OpenTrajectoryAnalysis/noctiluca
```

Developers
----------
Note the `Makefile`, which can be used to build the documentation (using
Sphinx); run unit tests and check code coverage; and build an updated package
for release with GNU `make`.

When editing the example notebooks,
[remember](https://nbsphinx.readthedocs.io/en/sizzle-theme/usage.html#Using-Notebooks-with-Git)
to remove output and empty cells before committing to the git repo.
[nbstripout](https://github.com/kynan/nbstripout) allows to do this
automatically upon commit.
