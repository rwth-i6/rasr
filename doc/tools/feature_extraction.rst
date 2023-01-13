Feature Extraction
==================

Audio file processing and the calculation of acoustic features is done using :doc:`Flow <../flow>` Networks.

To prevent calculating the features in every training step, recognition pass, or adaptation pass, the features are calculated beforehand and stored in caches.

The tool ``feature-extraction`` creates a flow network and feeds the audio files specified in the corpus to it.

**Configuration**:

The flow network definition for the feature extraction is specified in all applications by the selector ``feature-extraction``.

Parameters::

    file : the name of the flow file 
    network-file-path: the path where to find other flow files
