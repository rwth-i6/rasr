State tying
===========

CART file format
----------------

The CART implementation uses an XML-format. It begins with the XML-header

.. code-block :: xml

    <?xml version="1.0" encoding="ISO-8859-1"?>

The root element is named "decision-tree". It is followed by an element "properties-definition", containing several "key" and "valueMap"-elements.

.. code-block :: xml

    <decision-tree>
        <properties-definition> 
           <key>name of key</key>
           <value-map>
                <value id="0">value0</value>
                <value id="1">value1</value>
                ...
            </value-map>
            similarly the phoneme list 

Keys and ValueMaps are specified for the hmm-states, 

.. code-block :: xml

    <key>hmm-state</key>
    <value-map>
        <value id="0">0</value>
        <value id="1">1</value>
        <value id="2">2</value>
    </value-map>

the word position 

.. code-block :: xml

    <key>boundary</key>
    <value-map>
        <value id="1">begin-of-lemma</value>
        <value id="2">end-of-lemma</value>
        <value id="3">single-phoneme-lemma</value>
        <value id="0">within-lemma</value>
    </value-map>


and the phonemes for all positions in the allophone (history[0], central, future[0] for triphones)

.. code-block :: xml

    <key>history[0]</key>
        <value-map>
            <value id="0">#</value>
            <value id="1">E1</value>
    ...

The next element of the xml-file is named "questions". It contains the question-elements. In contrast to the legacy format all questions are listed and nothing is added automatically. The questions are specified for every allophone-position.

.. code-block :: xml

    <questions>
        <question description="silence">
            <key>central</key>
            <value>si</value>
         </question>
         <question description="CONSONANTS">
             <key>history[0]</key>
             <values>W _n b c ch d f g h j k l m n ng p q r s sh t v w x y z zh
             </values>
         </question>
         ...

Finally the tree is defined. The attribute "id" refers to the index of the question in case of a node or the the index of the cart class for leafs. Additional informations can be stored in the information-element. 

.. code-block :: xml

    <binary-tree>
        <node id="0">
           <information>
                <order>0</order>
                <size>83142165</size>
                <score>1.26782e+10</score>
            </information>
            <node id="4500">
                <information>
                    <order>1</order>
                    <size>9293883</size>
                    <score>1.27342e+09</score>
                </information>
            </node>
    ...

Lookup Table
============

Format per line: 
:ref:`Allophone State` followed by Mixture-ID

**Example** ::
    t{a+#}@f.1 593
    t{a+#}@f.2 796
    a{B+n}.0 1404
    a{B+n}.1 1594
    a{B+n}.2 1116
    n{a+s}.0 3311
    n{a+s}.1 3324
    n{a+s}.2 3258
    s{n+#}@f.0 202
    s{n+#}@f.1 260
    s{n+#}@f.2 72
