/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#ifndef _SPARSE_FLOW_VECTOR_HH
#define _SPARSE_FLOW_VECTOR_HH

/*! @todo rename this file to Vector.hh */

#include <algorithm>
#include <complex>
#include <string>
#include <vector>
#include <Core/BinaryStream.hh>
#include <Core/XmlStream.hh>
#include <Flow/Timestamp.hh>
#include <Sparse/SingleValSparseVector.hh>
namespace Sparse {

    /**
     * Block sparse vector with integration into the Flow network.
     *
     * Previosly: Flow::SparseVector
     * */
    template<class T>
    class Vector :
        public Flow::Timestamp, public SingleValueSparseVector<T> {

    private:
        typedef Vector<T> Self;
        typedef SingleValueSparseVector<T> Precursor;

//    protected:
//	using Precursor::v_;
//	using Precursor::InternalVector;
//	typedef typename Precursor::InternalVectorIterator InternalVectorIterator;
//	typedef typename Precursor::InternalVectorConstIterator InternalVectorConstIterator;


    public:
        using Precursor::begin;
        using Precursor::end;
        using Precursor::push_back;
        using Precursor::size;
        using typename Precursor::iterator;
        using typename Precursor::const_iterator;

        static const Flow::Datatype *type() {
            static Core::NameHelper<Self> name;
            static Flow::DatatypeTemplate<Self> dt(name);
            return &dt;
        };


        /** Creates a new sparse vector. */
        Vector() : Flow::Timestamp(type()), Precursor() {};


        /**
         * Creates a new sparse vector with given size.
         * @param size Initial size.
         */
        Vector(int size) : Flow::Timestamp(type()), Precursor(size) {}


        /**
         * Creates a new sparse vector with given size and content.
         * @param size Initial size
         * @param contentToFill Content to fill the sparse vector with.
         */
        Vector(int size, T contentToFill) : Flow::Timestamp(type()), Precursor(size, contentToFill) {}


        /**
         * Copy constructor
         * @param v Original data.
         */
        Vector(const Vector<T> &v) : Flow::Timestamp(type()), Precursor(v) {}


        /**
         * Copy constructor from Precursor.
         * @param v Original data.
         */
        Vector(const Precursor &v) : Flow::Timestamp(type()), Precursor(v) {}


        /** Destroys the sparse vector. */
        virtual ~Vector() { }

        /** Returns a pointer of a vector clone. */
        virtual Data* clone() const { return new Self(*this); }


        virtual Core::XmlWriter& dump(Core::XmlWriter &o) const ;
        virtual bool read(Core::BinaryInputStream &i);
        virtual bool write(Core::BinaryOutputStream &o) const ;
    };


    /**
     * Dumps the sparse vector to an XML stream.
     * @param o XML output stream to write to.
     */
    template <typename T>
        Core::XmlWriter& Vector<T>::dump(Core::XmlWriter &o) const {
        o  << Core::XmlOpen(datatype()->name())
            + Core::XmlAttribute("size", size())
            + Core::XmlAttribute("start", startTime())
            + Core::XmlAttribute("end", endTime());
        Precursor::dump(o);
        o << Core::XmlClose(datatype()->name());
        return o;
    }


    /**
     * Reads a sparse vector from a binary stream.
     * @param i Stream to read from.
     */
    template <typename T>
        bool Vector<T>::read(Core::BinaryInputStream &i){
        if (Precursor::read(i))
            return Flow::Timestamp::read(i);
        return false;
    }



    /**
     * Writes a sparse vector to a binary stream.
     * @param o Stream to write to.
     */
    template <typename T>
        bool Vector<T>::write(Core::BinaryOutputStream &o) const {
       if (Precursor::write(o))
           return Flow::Timestamp::write(o);
       return false;
    }



    /**
     * Outputs the sparse vector content to an XML stream.
     * @param o XML stream to dump on
     * @param v Vector to dump
     */
    template<class T> Core::XmlWriter& operator<< (Core::XmlWriter& o, const Vector<T> &v) {
        v.dump(o); return o;
    }


    /**
     * Writes the sparse vector to a binary output stream.
     * @param o Binary stream to write on.
     * @param v Vector to write.
     */
    template<class T> Core::BinaryOutputStream& operator<< (Core::BinaryOutputStream& o, const Vector<T> &v) {
        v.write(o); return o;
    }


    /**
     * Reads the sparse vector from a binary input stream.
     * @param i Binary stream to read from.
     * @param v Vector to store the read data in.
     */
    template<class T> Core::BinaryInputStream& operator>> (Core::BinaryInputStream& i, Vector<T> &v) {
        v.read(i); return i;
    }

} // namespace Flow


namespace Core {
    template <typename T>
    class NameHelper< Sparse::Vector<T> > : public std::string {
    public:
        NameHelper() : std::string(Core::NameHelper<Sparse::SingleValueSparseVector<T> >()) {}
    };
} // namespace Core



#endif // _FLOW_SPARSE_VECTOR_HH
