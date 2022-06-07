Mixture File
============

Standard Format
---------------

Mixture models produced during acoustic model training are by default stored as accumulators in a binary format.

The actual mixture model is automatically estimated from the read accumulators.

The binary format can be converted to an XML format (for debugging, visualization) using the AcousticModelTrainer.
However, the XML files cannot be converted to the binary format and are not readable by the tools.

.. code-block:: bash

    acoustic-model-trainer \
        --action=show-mixture-set \
        --*.mixture-set-trainer.old-mixture-set-file=my_mixture.mix \
        --config=/dev/null \
        --*.progress.channel=stderr \
        --*.log.channel=stdout \
        --*.warning.channel=stderr \
        --*.error.channel=stderr > my_mixure.xml

Mixture Text File Format
------------------------

aka "PMS". Structure of text mixture files (see also src/Mm/MixtureSet.cc : bool MixtureSet::read(std::istream&))

Note: see Mixture::read() in src/Mm/Mixture.cc for details on "version vs log-weight vs weight". ::

    #Version: 1.0
    #CovarianceType: DiagonalCovariance
    dim nMixtures nDensities nMeans nCovariances
    List of Mixtures: nDensities densityId-0 log(weight-0) densityId-1 log(weight-1) ...
    List of Densities: meanId CovarianceId
    List of Means: dim m1 m2 m3 ..
    List of Covariances: dim c1 w1 c2 w3 ...

**Usage**

Mixture set files using this format must have a filename ending in .pms or .gz.

Format Conversion
-----------------

Standard Format to Text Format

.. code-block:: bash

    acoustic-model-trainer \
      --action=convert-mixture-set \
      --*.mixture-set.file=binary-mixture-set.mix \
      --*.new-file=ascii:text-mixture.pms.gz

**Text Format to Standard Format**

.. code-block:: bash

    acoustic-model-trainer \
      --action=convert-mixture-set-to-mixture-set-estimator \
      --*.mixture-set-trainer.file=text-mixture.pms.gz \
      --*.new-mixture-set-file=binary-mixture-set.mix \
      --*.log.channel=stdout

Note that the generated mixture set estimator has weights instead of counts for each density. Applications using the convert mixture set have to set `*.minimum-observation-weight = 0` (see :doc:`training/mixture_set_estimation`).
