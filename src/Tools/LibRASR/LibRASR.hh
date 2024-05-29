#ifndef _TOOLS_LIBRASR_LIBRASR_HH
#define _TOOLS_LIBRASR_LIBRASR_HH

#include <string>
#include <vector>

#include <Core/Application.hh>

class _DummyApplication : Core::Application {
public:
    _DummyApplication();
    virtual ~_DummyApplication();

    int main(std::vector<std::string> const& arguments);
};

#endif