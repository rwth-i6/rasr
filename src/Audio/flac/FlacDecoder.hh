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
#ifndef FLACDECODER_HH_INCLUDED
#define FLACDECODER_HH_INCLUDED

#include <iostream>
#include <fstream>

extern "C" {
#include <FLAC/stream_decoder.h>
}

class FlacDecoder {
private:
    FILE* file_;
    FLAC__StreamDecoder *decoder_;

    struct FlacData {
        unsigned long int samplesToRead;
        unsigned long int samplesRead;
        unsigned long int samplePos;
        void *buffer;
        unsigned long int channels;
        unsigned long int bitsPerSample;
        unsigned long int sampleRate;
        unsigned long int totalSamples;
    } clientData_;

    static FLAC__StreamDecoderWriteStatus stream_decoder_write_callback_   (FLAC__StreamDecoder const*, FLAC__Frame const*, const FLAC__int32 * const *, void *);
    static void                           stream_decoder_metadata_callback_(FLAC__StreamDecoder const*, FLAC__StreamMetadata const*, void *);
    static void                           stream_decoder_error_callback_   (FLAC__StreamDecoder const*, FLAC__StreamDecoderErrorStatus, void *);

public:

    FlacDecoder() { }
    ~FlacDecoder() {
        FLAC__stream_decoder_delete(decoder_);
    }
  
    bool open(const char*);
  
    bool seek(unsigned long);
  
    unsigned long int getChannels() const;
    unsigned long int getBitsPerSample() const;
    unsigned long int getSampleRate() const;
    unsigned long int getTotalSamples() const;
  
    unsigned long int read(unsigned long int, void *);
};

#endif
