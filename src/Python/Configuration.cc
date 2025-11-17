#include "Configuration.hh"

#include <Core/Configuration.hh>

PyConfiguration::PyConfiguration()
        : Core::Configuration(), pythonSourceDescriptor_(db_->addSource("python", "N/A")) {
    setSelection("lib-rasr");
}

PyConfiguration::PyConfiguration(PyConfiguration const& c)
        : Core::Configuration(c), pythonSourceDescriptor_(db_->addSource("python", "N/A")) {
}

PyConfiguration::PyConfiguration(PyConfiguration const& c, std::string const& selection)
        : Core::Configuration(c, selection), pythonSourceDescriptor_(db_->addSource("python", "N/A")) {
}

void PyConfiguration::set(std::string const& name, std::string const& value) {
    Configuration::set(name, value, pythonSourceDescriptor_);
}
