#include "Configuration.hh"

#include <Core/Configuration.hh>

PyConfiguration::PyConfiguration()
        : Core::Configuration() {
    setSelection("lib-rasr");
}