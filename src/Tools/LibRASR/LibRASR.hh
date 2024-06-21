#ifndef _TOOLS_LIBRASR_LIBRASR_HH
#define _TOOLS_LIBRASR_LIBRASR_HH

#include <vector>
#include <string>

#include <Core/Application.hh>

class DummyApplication : public Core::Application {
public:
    DummyApplication();
    virtual ~DummyApplication();

    int main(std::vector<std::string> const& arguments);
};

#endif
