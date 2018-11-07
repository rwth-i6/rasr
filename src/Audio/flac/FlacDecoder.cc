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
#include <Core/Types.hh>
#include "FlacDecoder.hh"

FLAC__StreamDecoderWriteStatus FlacDecoder::stream_decoder_write_callback_(const FLAC__StreamDecoder *decoder_, const FLAC__Frame *frame, const FLAC__int32 * const buffer[], void *clientData) {
    if ( ((FlacData*) clientData)->samplesToRead >
         ((FlacData*) clientData)->samplesRead ) {
        unsigned long int blockSamples = frame->header.blocksize;
        unsigned long int channels = ((FlacData*) clientData)->channels;
        unsigned long int blockPos = 0;

        while( (((FlacData*) clientData)->samplesToRead >
                ((FlacData*) clientData)->samplesRead )
               && blockPos < blockSamples ) {
            for(unsigned int channel = 0; channel < channels; channel++) {
                ((short int*) ((FlacData*) clientData)->buffer)[(channels*(((FlacData*) clientData)->samplesRead))+channel] = (short int) buffer[channel][blockPos];
            }
            blockPos++;
            ((FlacData*) clientData)->samplesRead++;
        }
    }
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

void FlacDecoder::stream_decoder_metadata_callback_(const FLAC__StreamDecoder *decoder_,const FLAC__StreamMetadata *metadata, void *clientData) {
    return;
}

void FlacDecoder::stream_decoder_error_callback_(const FLAC__StreamDecoder *decoder_, FLAC__StreamDecoderErrorStatus status, void *clientData) {
    //std::cout << "ERROR: error callback" << std::endl;
    return;
}

bool FlacDecoder::open(const char* fileName) {
    file_ = fopen(fileName, "r");

    decoder_ = FLAC__stream_decoder_new();
    if (0 == decoder_) {
        std::cerr << "stream_decoder_new FAILED" << std::endl;
        return false;
    }

    if (!FLAC__stream_decoder_set_metadata_ignore_all(decoder_)) {
        //std::cout << "stream_decoder_set_metadata_ignore_all FAILED" << std::endl;
        return false;
    }

    if (!FLAC__stream_decoder_set_md5_checking(decoder_, false)) {
        //std::cout << "stream_decoder_set_md5_checking FAILED" << std::endl;
        return false;
    }

    clientData_.buffer        = nullptr;
    clientData_.samplesToRead = 0ul;
    clientData_.samplesRead   = 0ul;
    clientData_.samplePos     = 0ul;
    clientData_.channels      = 0ul;
    clientData_.bitsPerSample = 0ul;
    clientData_.sampleRate    = 0ul;
    clientData_.totalSamples  = 0ul;

    FLAC__StreamDecoderInitStatus status = FLAC__stream_decoder_init_FILE(decoder_, file_, stream_decoder_write_callback_, stream_decoder_metadata_callback_, stream_decoder_error_callback_, &clientData_);
    if (status != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
        //std::cout << "stream_decoder_init_FILE FAILED" << std::endl;
        return false;
    }

    if (!FLAC__stream_decoder_seek_absolute(decoder_, 0)) {
        //std::cout << "stream_decoder_seek_absolute FAILED" << std::endl;
        return false;
    }

    clientData_.channels      = FLAC__stream_decoder_get_channels(decoder_);
    clientData_.bitsPerSample = FLAC__stream_decoder_get_bits_per_sample(decoder_);
    clientData_.sampleRate    = FLAC__stream_decoder_get_sample_rate(decoder_);

    if (!FLAC__stream_decoder_process_until_end_of_metadata(decoder_)) {
        //std::cout << "stream_decoder_process_until_end_of_metadata FAILED" << std::endl;
        return false;
    }

    clientData_.totalSamples = FLAC__stream_decoder_get_total_samples(decoder_);

    return true;
}

bool FlacDecoder::seek(unsigned long int samplePos) {
    clientData_.samplePos = samplePos;
    return FLAC__stream_decoder_seek_absolute(decoder_, samplePos);
}

unsigned long int FlacDecoder::getChannels() const {
    return clientData_.channels;
}

unsigned long int FlacDecoder::getBitsPerSample() const {
    return clientData_.bitsPerSample;
}

unsigned long int FlacDecoder::getSampleRate() const {
    return clientData_.sampleRate;
}

unsigned long int FlacDecoder::getTotalSamples() const {
    return clientData_.totalSamples;
}

unsigned long int FlacDecoder::read(unsigned long int nSamples, void *buffer) {
    unsigned long int prevSamples = 0;
    clientData_.buffer = buffer;
    clientData_.samplesToRead = nSamples;
    clientData_.samplesRead = 0;

    FLAC__stream_decoder_seek_absolute(decoder_, clientData_.samplePos);
    while (prevSamples != clientData_.samplesRead) {
        prevSamples = clientData_.samplesRead;
        FLAC__stream_decoder_process_single(decoder_);
    }
    clientData_.samplePos += clientData_.samplesRead;
    unsigned long int samplesRead = clientData_.samplesRead;
    clientData_.samplesRead = 0;
    clientData_.samplesToRead = 0;
    return (samplesRead);
}
