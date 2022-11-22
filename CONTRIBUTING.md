Overview
--------

Coding conventions

- use clangformat
- camel case for files and directories (start with a capital)
- camel case for variables (classes: start with capital, everything else: start with lower case)
- avoid abbreviations
- avoid subdirectories
- private members should end with an underscore
- include guard should include folder name
- no exceptions


Code Organization
-----------------

**Rule**: File name extensions
 - C++ header files must have the extension ".hh". C++ implementation files must have the extension ".cc". 
---
**Rule**: An include file for a class should have a file name of the form "class name + extension".
 - Use uppercase and lowercase letters in the same way as in the source code.
---
**Rule**: Every include file must contain a mechanism that prevents multiple inclusions of the file. 
```
// $Id: ProgrammingGuidelines.html,v 1.10 2003/03/18 15:11:31 bisani Exp $
#ifndef _DIRECTORYNAME_FILENAME_HH
#define _DIRECTORYNAME_FILENAME_HH

// The rest of the file

#endif
```

Source Code Layout
------------------

**Rule**: 4 space indentation, no tabs
- All code lines must be properly indented according to their block nesting.
---
**Rule**: Special characters like page break must not be used.
 - Rationale: These characters are bound to cause problem for editors, printers, terminal emulators or debuggers when used in a multi-programmer, multi-platform environment. 
---
**Recommendation**: Logical units within a block should be separated by one blank line. 

---
**Rule**: Whenever a block exceeds one screen-full in length (about 70 lines) the closing brace should be commented. 
```
namespace Core {
    ...
} // namespace Core
```
---
**Rule**: Comments should be indented relative to their position in the code. 
```
while (true) {
    // Do something
    something();
}
```
NOT:
```
while (true) {
// Do something
    something();
}
```
