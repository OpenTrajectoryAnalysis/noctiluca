call append(0, ["Tutorials & Examples",
	       \"====================",
	       \"",
	       \".. toctree::",
	       \"   :maxdepth: 1",
	       \])
norm G
read !ls -1 examples/*.ipynb
g/^examples/norm I   
write! examples.rst
quit
