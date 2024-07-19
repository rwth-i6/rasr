"""Setup.py inspired by https://thomastrapp.com/posts/building-a-pypi-package-for-a-modern-cpp-project/"""
import os
from setuptools import setup, dist

PACKAGE_NAME = "librasr" # keyword to import
LIBRASR_SUBDIR = os.path.join("src", "Tools", "LibRASR")
LIBRASR_SO = "librasr.so"

assert os.path.exists(os.path.join(LIBRASR_SUBDIR, LIBRASR_SO)), \
    "Could not find librasr shared library. Did you compile it correctly?"
class BinaryDistribution(dist.Distribution):
    """Helper class: We assume that librasr.so has already been compiled."""
    def has_ext_modules(self) -> bool:
        return True

# get README content for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README')) as f:
    long_description = f.read()

setup(
    name='RASR',

    # include shared library and link to lower case package name
    packages=[PACKAGE_NAME],
    package_dir={PACKAGE_NAME: LIBRASR_SUBDIR},
    package_data={PACKAGE_NAME: [LIBRASR_SO]},
    include_package_data=True,

    description="RASR as a python module.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # use custom distribution class to export librasr.so
    distclass=BinaryDistribution,

    version='0.0.1',
    url='https://github.com/rwth-i6/rasr',
    author_email='rwthasr@i6.informatik.rwth-aachen.de',
)
