This branch is dedicated for the integration of the generic seq-to-seq decoder into RASR master (https://github.com/rwth-i6/rasr/pull/45).
The decoder is developed by Wei Zhou during his PhD study before RASR github version was created.
For simplicity, the complete decoder is adapted and rebased onto the master branch in one shot for
various testings using differernt seq-to-seq models, LMs, label units/topologies, NN architectures
and search settings, etc.

For citation and more details of this decoder, please refer to the following paper:
@InProceedings {zhou:rasr2:interspeech2023,
author= {Wei Zhou and Eugen Beck and Simon Berger and Ralf Schl\"uter and Hermann Ney},
title= {{RASR2: The RWTH ASR Toolkit for Generic Sequence-to-sequence Speech Recognition}},
booktitle= {Interspeech},
pages= {4094-4098},
year= 2023,
month= Aug
}

Example configs related to the experiments in the paper can be found in the 'examples' folder.

--------------------------------------------------------------------------------------------

Sprint - The RWTH Speech Recognition Framework

* Requirements
  Debian package name given in brackets
  - GCC >= 4.8 (gcc, g++)
  - GNU Bison  (bison)
  - GNU Make   (make)
  - libxml2    (libxml2, libxml2-dev)
  - libsndfile (libsndfile1, libsndfile1-dev)
  - libcppunit        (libcppunit, libcppunit-dev)
  - LAPACK     (lapack3, lapack3-dev)
  - BLAS       (refblas3, refblas3-dev)

* Build
  - Adapt Config.make, Options.make, config/os-linux.make to your environment
  - Check requirements:
    ./scripts/requirements.sh
  - Compile
    make
  - Install: Will install executables in arch/linux-intel-standard/
    (INSTALL_TARGET in Options.make)
    make install

* Documentation
  - Code Documentation (requires Doxygen)
    Run 'doxygen' in src/
    Documentation is generated in src/api/html
  - http://www-i6.informatik.rwth-aachen.de/rwth-asr
    Wiki: http://www-i6.informatik.rwth-aachen.de/rwth-asr/manual
  - Manual: http://www-i6.informatik.rwth-aachen.de/sprintdoc
    Login on request

* Signal Analysis
  Flow files for common acoustic features:
  http://www-i6.informatik.rwth-aachen.de/rwth-asr/files/flow-examples-0.2.tar.gz 


--
(c) 2000-2020 RWTH Aachen University, Lehrstuhl fuer Informatik 6
http://www-i6.informatik.rwth-aachen.de/rwth-asr/
rwthasr@i6.informatik.rwth-aachen.de
