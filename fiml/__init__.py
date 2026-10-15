# FROM THE __init__.py OF THE ENTIRE PACKAGE (WITH SUBPACKAGES):
# When importing a module with
#    import myPackage
# Python will only bring the TOP-LEVEL package into the current namespace;
# NOT the entire package hierarchy. All the subpackages (i.e. the directories
# inside the package directory) and the modules (i.e. the .py files) WON'T be
# imported, therefore a statement like
#    z = myPackage.myModule1.myCode1.my_example_function_1(x)
# won't work because Python won't see any "myModule1" subpackage

# To fix the problem, you have 2 solutions:
# 1) EITHER import each MODULE separately
#	import myPackage.myModule1.myCode1 as myC1
#	z = myC1.my_example_function_1(x)
# 2) OR add some import statements into the __init__.py files, so that Python
# will do all the imports under the hood.
# In particular:
# a) import the SUBPACKAGES from withing the top-level
#from . import myModule1
#from . import myModule2
# b) and then import the MODULES from within each subpackage
# (see all the other __init__.py files in the subdirectories of this package)

# FROM THE __init__.py OF THE THE SINGLE SUBPACKAGE:
# For the explaination of the these import statements, see the __init__.py
# file in the top-level directory of the package.
# In short: each __init__.py import the subpackages (i.e. the subdirectories
# contained in the parent directory of the __init__.py) in a "cascade", until
# yuo reach the lowest level where no more subpackages exist.
# At that level, import each module (i.e. the single .py files).
# In this way, you'll be able to import the entire package (with all its
# subpackages and modules promptly available) with just
#	import myPackage

from . import auxiliaries
from . import fiml
from . import parallel
from . import main


__version__ = '0.0.1'
__author__ = ['Andrea Pinardi']
__email__ = ['andrea.pinardi@polimi.it']