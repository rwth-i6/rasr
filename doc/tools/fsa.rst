FSA tool
========

FSA is an efficient and flexible toolkit to create, manipulate and optimize finite-state automata. It supports weighted and unweighted automata as well as transducers and is designed to use on-demand computations in order to keep memory footprint as low as possible. It may be used to model various problems from the field of natural language processing.

**Usage**::

    fsa [OPTION(S)] <FILE | OPERATION> ...

options:

| ``--help`` : show help
| ``--progress=yes`` : show progress during operations
| ``--resources=yes`` : print resource database

The tool maintains a stack of automata. Passing a file will put the automaton on top of the stack. Operations will take the automata from the top of the stack, and push the result back on top of it.

Operations
----------

**Algorithms**

parameters and defaults in brackets, use e.g. closure,kleene or nbest,n=100

| ``best`` : extract [n(1)] best path(s) 
| ``closure`` : [kleene] closure of the topmost automaton 
| ``collect`` : collect each arc weight and [value] 
| ``concat`` : concat the [n(2)] topmost automata 
| ``complement`` : automaton that represents the complement language 
| ``compose`` : compose the two topmost automata [filter=(match),seq] 
| ``determinize`` : determinize topmost automaton [disambiguate] 
| ``difference`` : build the difference of the topmost and second topmost automaton 
| ``duplicate`` : duplicate topmost automaton 
| ``extend`` : extend each arc weight by [value] 
| ``expm`` : weight --> exp(-weight) 
| ``invert`` : swap input and output labels 
| ``levenshtein`` : calculates the levenshtein distance of the two topmost automata 
| ``map-input`` : map input labels using output alphabet of second topmost automaton 
| ``map-output`` : map output labels using input alphabet of second topmost automaton 
| ``multiply`` : multiply each arc weight by [value] (log and tropical semiring only) 
| ``minimize`` : minimize topmost automaton 
| ``normalize`` : normalizes state ids of topmost automaton (i.e. initial = 0, no gaps) 
| ``partial`` : partial automaton starting at state [id] 
| ``permute`` : permute automaton with a window of [n=(infinity)], [type=(ibm),inv,itg,local], 
| ``[prob=(0.0)]`` : OR [dist=(0.0)] with a maximum distortion of [max=(dist> 0 ? 20 : infinity)] 
| ``posterior`` : calculate arc posterior weights 
| ``posterior64`` : calculate arc posterior weights (numerically more stable version for log semiring) 
| ``posteriorE`` : calculate arc posterior weights with expectation semiring 
| ``project`` : project [type=(input),output] labels to input labels 
| ``prune`` : prune arcs using path posterior weights [beam] threshold 
| ``push`` : push weights [to=(final),initial] state 
| ``random`` : select a random path 
| ``remove`` : remove [type=(epsilons),disambiguators] from topmost automaton 
| ``remove`` : remove arcs with disambiguation symbols or replace by epsilons 
| ``sort`` : sort all edges by [(arc),input,output,weight] 
| ``sync-prune`` : prune states using synchronuous state potentials and [beam] threshold 
| ``transpose`` : reverse the direction of all arcs 
| ``trim`` : removes all but the connected and disconnected states 
| ``unite`` : unite the [n(2)] topmost automata 
| ``fuse`` : fuse initial states of the [n(2)] topmost automata 

**Output**

| ``draw`` : write topmost automaton to [file=(-)] in dot format [best,detailed] 
| ``write`` : write topmost automaton or [input,output] alphabet (both input and output is possible, too) or only the states [states] to [file=(-)] 

**Control**

| ``cache`` : caches states of topmost transducer 
| ``copy`` : creates a static copy of the topmost transducer 
| ``delete`` : delete topmost transducer 
| ``default`` : set the default semiring for all following read operations (see list below) 
| ``semiring`` : change the semiring of the topmost automaton (see list below) 

**Diagnostics**

| ``count`` : [(input),output] arc count statistics for [label] or number of [paths] 
| ``info`` : print sizes of topmost automaton 
| ``memory`` : print detailed memory info of topmost automaton 
| ``time`` : print time consumed by preceeding operation 
| ``wait`` : wait for pressing <ENTER> 

Semirings
---------

count, log, probability, tropical, tropical-integer

tolerance=(1) for log

File formats
------------
prepend ``att:`` / ``bin:`` / ``lin:`` / ``xml:`` / ``trxml:`` in order to select file format

| ``packed:`` for compressed storage
| ``combine`` to combine automata from different files

Examples
--------

**Plotting a binary fsa**::

    fsa bin:input.binfsa draw | dot -Tsvg > out.svg

**Compute the top n-best paths in an fsa**::

    fsa input.fsa best,n=5 write,file=out.fsa

**Merge alphabet FSA into other FSA**

This is not directly supported, but we can make use of the unite operation::

    fsa bin:segment-am.binfsa bin:input-alphabet.binfsa unite trim draw,file=fsa.dot
