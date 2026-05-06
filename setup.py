"""Build the Python package from the CMake librasr target."""

import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

import numpy
from setuptools import Extension, dist, setup
from setuptools.command.build_ext import build_ext

PACKAGE_NAME = "librasr"  # keyword to import
LIBRASR_SUBDIR = Path("src") / "Tools" / "LibRASR"


class CMakeExtension(Extension):
    """A setuptools extension that is built by CMake."""

    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    """Build librasr with CMake and copy the extension into the package."""

    def build_extension(self, ext: CMakeExtension) -> None:
        output_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        source_dir = Path(__file__).parent.resolve()
        build_dir = Path(
            os.environ.get("CMAKE_BUILD_DIR", Path(self.build_temp) / "cmake-build")
        ).resolve()

        if self.dry_run:  # type: ignore
            return

        artifact = self._build_librasr(source_dir, build_dir, output_dir)

        if artifact.resolve() != output_path:
            shutil.copy2(artifact, output_path)

    def _build_librasr(
        self,
        source_dir: Path,
        build_dir: Path,
        output_dir: Path,
    ) -> Path:
        # CMake setup
        build_dir.mkdir(parents=True, exist_ok=True)
        config = (
            "debug" if self.debug else os.environ.get("CMAKE_BUILD_TYPE", "standard")
        )

        cmake_args = [
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_NumPy_INCLUDE_DIR={numpy.get_include()}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
        ]
        cmake_args.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))

        subprocess.check_call(["cmake", *cmake_args])

        # CMake build
        build_args = [
            "--build",
            str(build_dir),
            "--target",
            "librasr",
            "--parallel",
            str(os.cpu_count() or 1),
        ]
        build_args.extend(shlex.split(os.environ.get("CMAKE_BUILD_ARGS", "")))
        subprocess.check_call(["cmake", *build_args])

        return self._find_librasr_artifact(output_dir)

    def _find_librasr_artifact(self, output_dir: Path) -> Path:
        candidates = list(output_dir.glob("librasr*.so"))

        if not candidates:
            raise RuntimeError(
                f"Could not find the CMake-built librasr Python extension in {output_dir}."
            )

        return max(candidates, key=lambda path: path.stat().st_mtime)


class BinaryDistribution(dist.Distribution):
    """Mark the package as platform-specific because it contains a binary module."""

    def has_ext_modules(self) -> bool:
        return True


# get README content for long description
this_directory = Path(__file__).parent.resolve()
with open(this_directory / "README") as f:
    long_description = f.read()

setup(
    name="RASR",
    # include shared library and link to lower case package name
    packages=[PACKAGE_NAME],
    package_dir={PACKAGE_NAME: str(LIBRASR_SUBDIR)},
    ext_modules=[CMakeExtension(f"{PACKAGE_NAME}.{PACKAGE_NAME}")],
    cmdclass={"build_ext": CMakeBuild},
    description="RASR as a python module.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "tensorflow"],
    # use custom distribution class to mark the wheel as platform-specific
    distclass=BinaryDistribution,
    version="0.0.1",
    url="https://github.com/rwth-i6/rasr",
    author_email="rwthasr@i6.informatik.rwth-aachen.de",
)
