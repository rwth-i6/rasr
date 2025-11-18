#ifndef _PYTHON_CONFIGURATION_HH
#define _PYTHON_CONFIGURATION_HH

#include <Core/Configuration.hh>

class PyConfiguration : public Core::Configuration {
public:
    PyConfiguration();
    PyConfiguration(PyConfiguration const& c);
    PyConfiguration(PyConfiguration const& c, std::string const& selection);

    void set(std::string const& name, std::string const& value = "true");

private:
    SourceDescriptor const* pythonSourceDescriptor_;
};

#endif  // _PYTHON_CONFIGURATION_HH
