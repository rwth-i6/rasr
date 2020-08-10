/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef _FLOW_ARCHIVEWRITER_HH
#define _FLOW_ARCHIVEWRITER_HH

#include <Core/Archive.hh>
#include "Attributes.hh"
#include "DataAdaptor.hh"

namespace Flow {

template<typename T>
struct ArchiveWriter {
    Core::Archive*            archive_;
    Flow::DataAdaptor<T>*     data_;
    Flow::DataPtr<Flow::Data> dataPtr_;  // owns data_
    ArchiveWriter(Core::Archive* archive)
            : archive_(archive),
              data_(new Flow::DataAdaptor<T>()),
              dataPtr_(data_) {}
    void write(const std::string& filename) {
        // See Flow::CacheWriter().
        // Compatible with Flow cache nodes.

        {
            // Write attribs, i.e. datatype.
            Core::ArchiveWriter w(*archive_, filename + ".attribs", true);
            Core::XmlWriter     xw(w);
            Flow::Attributes    attributes;
            attributes.set("datatype", data_->datatype()->name());
            xw << attributes;
        }

        {
            // Write data.
            Core::ArchiveWriter w(
                    *archive_, filename, /*compress*/ true);
            Core::BinaryOutputStream b(w);
            b << data_->datatype()->name();
            b << 1;  // see Datatype::writeGatheredData()
            data_->datatype()->writeData(b, dataPtr_);
        }
    }
};

}  // namespace Flow

#endif  // ARCHIVEWRITER_HH
