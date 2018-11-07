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
#ifndef _SIGNAL_FIR_FILTER_HH
#define _SIGNAL_FIR_FILTER_HH

#include <Core/BinaryStream.hh>
#include "Window.hh"
#include "KaiserWindowFunction.hh"
#include "Convolution.hh"

namespace Signal
{

    // WindowingFirFilter
    /////////////////////

    class WindowingFirFilter : public WindowBuffer {
    public:

        typedef WindowBuffer Predecessor;
        typedef WindowBuffer::Time Time;
        typedef WindowBuffer::Sample Sample;

    private:

        /** length of output vectors
         */
        Time lengthInS_;

        KaiserWindowFunction window_;

        /** executes convolution between the filter and the imput signal
         *
         *  -remark: the number of FFT points used for
         *   calculating the convolution is given by length of window.
         */
        Convolution convolution_;

        /** gain, frequency pairs */
        std::vector<std::pair<f64, f64> > cutOff_;
        /** max amplitude oscillation in pass band and in stop band interval */
        f64 overshoot_;
        /** width of transition region in Hz */
        f64 transitionRegionWidthInHz_;
        /** (length - 1) of filter response */
        u32 M_;

        /** if true output signal has same size as the input one
         *  if no output signal is extended by M_ samples at the beginning and at the end.
         */
        bool removeDelay_;

    private:

        void getFilterResponse(std::vector<Sample> &h);

        u32 delay() const { return M_ / 2; }

        /** Beta parameter for Kaiser window
         *    take form Oppenheimer-Schafer: Discrete Time Signal Processing,
         *    chapter Yhe Kaiser Window Filter Design Method
         *
         * @param overshoot: max amplitude oscillation in pass band and in stop band interval
        */
        static f64 getBeta(f64 overshoot);

        /** M (length - 1) parameter for Kaiser window
         *    take form Oppenheimer-Schafer: Discrete Time Signal Processing,
         *    chapter Yhe Kaiser Window Filter Design Method
         *
         * @param overshoot: max amplitude oscillation in pass band and in stop band interval
         * @param deltaOmega: width of transition region in relitive omega unit
         */
        static u32 getM(f64 overshoot, f64 deltaOmega);

    protected:

        virtual void init();

        virtual void transform(Flow::Vector<Sample> &out);

    public:

        WindowingFirFilter();
        virtual ~WindowingFirFilter() {}

        void setSampleRate(f64 sampleRate);

        /** cut-off: gain-omega) pairs */
        void setCutOff(const std::vector<std::pair<f64, f64> >& cutOff);

        /** max amplitude oscillation of the filter response
         *    in pass band and in stop band interval
         */
        void setOvershoot(f64 overshoot);

        /** width of transition region in Hz */
        void setTransitionRegionWidthInHz(f64 transitionRegionWidthInHz);

        /** sets length of output vectors */
        void setLengthInS(Time length);
        /** length of output vectors */
        Time lengthInS() const { return lengthInS_; }

        /** if true output signal has same size as the input one
         *  if no output signal is extended by M_ samples at the beginning and at the end.
         */
        void setRemoveDelay(bool remove) { removeDelay_ = remove; }
        bool removeDelay() const { return removeDelay_; }
    };


    // WindowingFirFilterParameter
    //////////////////////////////

    class WindowingFirFilterParameter : public Flow::Timestamp {
        typedef Flow::Timestamp Precursor;
        typedef WindowingFirFilterParameter Self;
    private:
        std::vector<std::pair<f64, f64> > cut_off_;
    public:
        static const Flow::Datatype *type() {
            static Flow::DatatypeTemplate<Self> dt("windowing-fir-filter-parameter");
            return &dt;
        }
        WindowingFirFilterParameter() : Precursor(type()) {}
        virtual ~WindowingFirFilterParameter() {}

        virtual Flow::Data* clone() const { return new WindowingFirFilterParameter(*this); };

        virtual Core::XmlWriter& dump(Core::XmlWriter &o) const;
        bool read(Core::BinaryInputStream &i);
        bool write(Core::BinaryOutputStream &o) const;

        std::vector<std::pair<f64, f64> >& getCutOff() { return cut_off_; };
    };


    /** WindowingFirFilterNode
     */
    class WindowingFirFilterNode : public SlidingAlgorithmNode<WindowingFirFilter> {
    public:

        typedef SlidingAlgorithmNode<WindowingFirFilter> Predecessor;

    private:

        static Core::ParameterFloat paramLength;
        static Core::ParameterBool paramRemoveDelay;
        static Core::ParameterString paramCutOff;
        static Core::ParameterFloat paramOvershoot;
        static Core::ParameterFloat paramTransitionRegionWidth;

    private:

        bool parseCutOff(const std::string& value, std::vector<std::pair<f64, f64> >& cutOff);

    public:

        static std::string filterName() { return "signal-windowing-fir-filter"; }

        WindowingFirFilterNode(const Core::Configuration &c);
        virtual ~WindowingFirFilterNode() {}

        virtual bool setParameter(const std::string &name, const std::string &value);
        virtual bool configure();
        virtual Flow::PortId getInput(const std::string &name);

        virtual bool work(Flow::PortId p);
    };

} // namespace Signal

#endif // _SIGNAL_FILTER_HH
