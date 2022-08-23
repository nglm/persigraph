from setuptools import setup, find_packages

PACKAGE_NAME = "persigraph"
VERSION = '0.0.1'
DESCRIPTION = "Python implementation of the PersistentGraph method"
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    # Required
    # the name must match the folder name "package_name"
    name=PACKAGE_NAME,
    # Required
    version=VERSION,
    author='Natacha Galmiche',
    author_email='natacha.galmiche@uib.no',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/nglm/' + PACKAGE_NAME,
    license='MIT',
    # Required
    # You can just specify package directories manually here
    # (e.g. packages=['toolbox']) if your project is simple.
    # Or you can use find_packages().
    packages=find_packages(where=PACKAGE_NAME),
    # Specify what dependencies a project minimally needs to run
    install_requires=[],
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={},
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],  # Optional
)