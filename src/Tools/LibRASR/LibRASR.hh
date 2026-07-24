#ifndef _TOOLS_LIBRASR_LIBRASR_HH
#define _TOOLS_LIBRASR_LIBRASR_HH

#include <string>
#include <vector>

#include <Core/Application.hh>

class DummyApplication : public Core::Application {
public:
    DummyApplication();
    virtual ~DummyApplication();

    void initLogging(Core::Configuration const& loggingConfig);

    int main(std::vector<std::string> const& arguments);
};

#endif
