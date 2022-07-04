Bliss Corpus
============

A corpus file defines the audio data used and provides the transcription and the segmentation.

The Bliss corpus description format is an XML document. The recommended filename extension is “.corpus”.

Terminology
-----------

recording
    continuous stream of audio data, an audio file 
corpus
    collection of several recordings 
segment
    a contiguous (and preferably homogenous) part of a recording. 
speaker
    a person who speaks 
transcription
    textual representation of speech (or other acoustic) events; usually orthographic 

Structure
---------

The root element of a corpus file is always a ``<corpus>`` element. Each corpus has a symbolic name given by the name attribute, e.g. ``<corpus name="EPPS">``

**Logical Inclusion**:

Corpora often contain large numbers of recordings, so that it is convenient to structure a corpus into smaller entities. Inside a corpus, subcorpora can be defined using the <subcorpus> element. Since a subcorpus is a corpus, this technique can be applied recursively. Subcorpora also have symbolic names

**Physical Inclusion**:

It is often impratical to store the whole corpus in a single file. For this reason a <include> element can be used to insert the contents of another file at the current position. This phyisical inclusion is flat, i.e. the included corpus is part of the current one and not a sub-corpus of it. (Of course <include> can be combined with <subcorpus> to achieve this effect.)

**Example**::

    <?xml version="1.0" encoding="ISO-8859-1"?>
    <!DOCTYPE corpus SYSTEM "corpus.dtd">
    <corpus name="coretex-de">
      <include file="speakers.corpus"/>
      <subcorpus name="radio">
        <include file="WDR.corpus"/>
        <include file="SWR.corpus"/>
        <include file="DLR.corpus"/>
      </subcorpus>
      <subcorpus name="tv">
        <include file="ARD.corpus"/>
        <include file="ZDF.corpus"/>
        <include file="NTV.corpus"/>
      </subcorpus>
    </corpus>

Named Entities and Addressing
-----------------------------

It is often necessary to refer to a certain part of a corpus. To allow this, most items defined in a corpus have a symbolic name which is specified using the name attribute of their respective XML element. We will call these named corpus entity. The logical structure of a corpus is tree-like, i.e. there is parent-child relationship between the items. The full name of a named entity is obtained by prefixing its name with the full name of its parent separated by a slash ("/") (this is analogous to an absolute path file in a UNIX file system). The full name can be used to address an entity globally. E.g. a file with time alignment paths will include the full names of each segment. A list of VTN warping factors can refer to speakers by the full names of the speaker declaration. Within the corpus file only "upward" addressing is possible, i.e. only entities defined in ancestor entities can be referenced.

Recordings
----------

A recording is a continuous stream of audio data. An audio file (WAV, MPEG, FLAC, Sphere, ...) is a recording. A recording can contain a complete radio show, a dialog of several persons, a single sentence or just a single word. A recording may consist of several tracks. E.g. left/right of a stereo recoding, both sides of a telephone conversation, original- / lip-sync-version of a movie, ... In the corpus file, recordings are represented by ``<recording>`` elements. They are named corpus entities just as corpora. The audio attribute is used to specify the respective audio file, which is not related to the recording’s full name. A recording can only be child of a corpus or a subcorpus. The attribute name needs to be unique across the corpus.

Segments
--------

A segment is a (usually) short part of a recording. Segments are contiguous, they have a defined start and ending time measured in seconds relative to the beginning of the recording. Note that a segment does not necessarily contain speech. A segment can have a specific type, e.g. speech, music, silence, ... It is not required that a recording is seamlessly covered by its segments. Segments may overlap and there can be regions of a recording which are not covered by any segment. Segments are described by ``<segment>`` elements, which can only occur within a recording. Segments are named corpus entities. The attributes start and end are use to define the start and ending time.

Speakers
--------

In the context of Bliss the term “speaker” always refers to the original producer of a speech signal, which is usually a human being. A speaker is described using a ``<speaker-description>`` element. Usually there is a one-to-one correspondence between speaker description elements and speaking persons in the corpus. A ``<speaker>`` element is used to reference a previously defined speaker. Speech segments may have a speaker associated with them via the <speaker> element. Corpora and recordings may have default speakers. The contents of a speaker description are not fully specified. (See for a description of which elements are understood by the current implementation.) The idea is that each corpus may have its own extensions, so that all information available is preserved. The following list suggests element names to use for typically relevant information:
* ``<gender>`` (“male” or “female”)
* ``<name>`` real name
* ``<age>`` or ``<date-of-birth>`` (The latter is preferred.)
* ``<native-language>`` or ``<native>`` (“yes” or “no”) 

**Example**::

    <?xml version="1.0" encoding="ISO-8859-1"?>
    <!DOCTYPE corpus SYSTEM "corpus.dtd">
    <corpus name="coretex-de">
      <speaker-description name="UW">
        <name>Ulrich Wickert</name>
        <gender>male</gender>
      </speaker-description>
      <recording name="tagesthemen" audio="ARD-20011116-2245.mp3">
        <segment>
          <speaker name="UW"/>
          <orth>Guten Abend ...</orth>
        </segment>
      </recording>
    </corpus>

Transcriptions
--------------

A transcription is a textual representation of a (speech) segment. Most of the time orthographic transcriptions are used, which are denoted by an ``<orth>`` element. It is recommended to use the conventional orthographic rules of the respective language. Non-ASCII characters must be represented in the encoding delared in the XML header. By convention words without proper orthographic form should be denoted with square brackets. This applies to: silence, non-speech events such as hesitation and background noise, and special LM tokens. This is useful, in particular, to distinguish a word from the event it describes. E.g. [breath] indicates that there is audible breath noise, while “breath” means that the speaker is actually talking about respiration.


File Format Specification
-------------------------

<corpus>
""""""""

* Description: root element
* Allowed contexts: xml root element
* Attributes:

  * ``name`` (required) If the corpus description file is included from another corpus, the corpus name must match the name of the corpus into which it is included.

<subcorpus>
"""""""""""

* Description: define a sub-corpus, a part of a corpus
* Allowed contexts: ``<corpus>``, ``<subcorpus>``
* Attributes:

  * ``name`` (required)

<include>
"""""""""
* Description: include another corpus description file as part of the current corpus
* Allowed contexts: ``<corpus>``, ``<subcorpus>``
* Attributes:

  * ``file`` (required) The attribute value is first resolved using the configuration of the corpus description object. The result is then intepreted as a path relative to the corpus file.

<recording>
"""""""""""

* Description: describes a recording
* Allowed contexts: ``<corpus>``, ``<subcorpus>``
* Attributes:

  * ``name`` (required) needs to be unique
  * ``audio`` (required) name of the corresponding audio file. The attribute value is first resolved using the configuration of the corpus description object. The result is then interpreted as a path relative to a configurable base directory. By default the directory of the corpus file is used.

<segment>
"""""""""

* Description: describes a segment (part of a recording).
* Allowed contexts: ``<recording>``
* Attributes:

  * ``name`` (optional) If no name is given, the segments within a recording will be numbered starting with one. Note that in this case the order in which segments are specified is significant and must not be changed. Named and unnamed segments should not be mixed within the same recording.
  * ``start``, ``end`` (required) Start and end time of a segment.
  * ``track`` select a particular track of a multi-track recording

<orth>
""""""

* Description: orthographic transcription of a segment (if appropriate and known)
* Allowed contexts: ``<segment>``

<condition-description>
"""""""""""""""""""""""

* Description: Define a new acoustic recording condition and describe it.
* Allowed contexts: ``<corpus>``,  ``<subcorpus>``,  ``<recording>``,  ``<segment>``
* Attributes:

  * ``name`` (optional) If name is not given, an annonymous condition is defined, which is the default for the enclosing corpus section.

<condition>
"""""""""""

* Description: Identify the acoustic condition of a segment or select a default.
* Allowed contexts: ``<corpus>``, ``<recording>``, ``<segment>``
* Attributes:

  * ``name`` (required) identifier of an acoustic condition defined previously (using ``<condition-description>``)

<speaker-description>
"""""""""""""""""""""

* Description: Create a new speaker definition and describe the speaker.
* Allowed contexts: ``<corpus>``, ``<subcorpus>``, ``<recording>``, ``<segment>``
* Attributes: ``name`` (optional) If name is not given, an annonymous speaker is defined, which is the default speaker of the enclosing corpus section.

<speaker>
"""""""""

* Description: Identify the speaker who produced an utterance or select a default speaker.
* Allowed contexts: ``<corpus>``, ``<subcorpus>``, ``<recording>``, ``<segment>``
* Attributes:

  * ``name`` (required) identifier of a previously defined speaker (using ``<speaker-description>``)

<name>
""""""

* Description: real name of a speaker
* Allowed contexts: ``<speaker-description>``

<gender>
""""""""

* Description: gender of a speaker
* Allowed contexts: ``<speaker-description>``

<age>
"""""

* Description: age of a speaker
* Allowed contexts: ``<speaker-description>``

<native-language>
"""""""""""""""""

* Description: native language of a speaker
* Allowed contexts: ``<speaker-description>``

See also
--------

:ref:`Common component configuration <Corpus Configuration>`
