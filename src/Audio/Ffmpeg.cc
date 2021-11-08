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
#include "Ffmpeg.hh"

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

namespace {

Flow::Timestamp* alloc_buffer(AVSampleFormat fmt) {
    switch (fmt) {
        case AV_SAMPLE_FMT_U8: return new Flow::Vector<u8>();
        case AV_SAMPLE_FMT_S16: return new Flow::Vector<s16>();
        case AV_SAMPLE_FMT_S32: return new Flow::Vector<s32>();
        case AV_SAMPLE_FMT_FLT: return new Flow::Vector<f32>();
        case AV_SAMPLE_FMT_DBL: return new Flow::Vector<f64>();
        default: return nullptr;
    }
}

template<typename T>
size_t add_to_buffer(Flow::Vector<T>* buffer, AVFrame const* frame, u32 sample_offset = 0u) {
    size_t old_size            = buffer->size();
    size_t num_total_samples   = frame->channels * frame->nb_samples;
    size_t total_sample_offset = frame->channels * sample_offset;
    if (num_total_samples > total_sample_offset) {
        size_t added_total_samples = num_total_samples - total_sample_offset;
        buffer->resize(buffer->size() + added_total_samples);
        T* out = buffer->data() + old_size;
        memcpy(out, frame->data[0] + total_sample_offset * sizeof(T), added_total_samples * sizeof(T));
    }
    return buffer->size();
}

size_t add_to_buffer(Flow::Timestamp* buffer, AVFrame const* frame, AVSampleFormat fmt, u32 sample_offset = 0u) {
    switch (fmt) {
        case AV_SAMPLE_FMT_U8: return add_to_buffer<u8>(dynamic_cast<Flow::Vector<u8>*>(buffer), frame, sample_offset);
        case AV_SAMPLE_FMT_S16: return add_to_buffer<s16>(dynamic_cast<Flow::Vector<s16>*>(buffer), frame, sample_offset);
        case AV_SAMPLE_FMT_S32: return add_to_buffer<s32>(dynamic_cast<Flow::Vector<s32>*>(buffer), frame, sample_offset);
        case AV_SAMPLE_FMT_FLT: return add_to_buffer<f32>(dynamic_cast<Flow::Vector<f32>*>(buffer), frame, sample_offset);
        case AV_SAMPLE_FMT_DBL: return add_to_buffer<f64>(dynamic_cast<Flow::Vector<f64>*>(buffer), frame, sample_offset);
        default: return 0ul;
    }
}

template<typename T>
Flow::Timestamp* create_output_buffer(Flow::Vector<T>* buffer, u32 numTotalSamples) {
    size_t           output_size = std::min<size_t>(buffer->size(), numTotalSamples);
    Flow::Vector<T>* result      = new Flow::Vector<T>(output_size);
    std::copy(buffer->begin(), buffer->begin() + output_size, result->begin());
    std::copy(buffer->begin() + output_size, buffer->end(), buffer->begin());
    buffer->resize(buffer->size() - output_size);
    return result;
}

Flow::Timestamp* create_output_buffer(Flow::Timestamp* buffer, u32 numTotalSamples, AVSampleFormat fmt) {
    switch (fmt) {
        case AV_SAMPLE_FMT_U8: return create_output_buffer<u8>(dynamic_cast<Flow::Vector<u8>*>(buffer), numTotalSamples);
        case AV_SAMPLE_FMT_S16: return create_output_buffer<s16>(dynamic_cast<Flow::Vector<s16>*>(buffer), numTotalSamples);
        case AV_SAMPLE_FMT_S32: return create_output_buffer<s32>(dynamic_cast<Flow::Vector<s32>*>(buffer), numTotalSamples);
        case AV_SAMPLE_FMT_FLT: return create_output_buffer<f32>(dynamic_cast<Flow::Vector<f32>*>(buffer), numTotalSamples);
        case AV_SAMPLE_FMT_DBL: return create_output_buffer<f64>(dynamic_cast<Flow::Vector<f64>*>(buffer), numTotalSamples);
        default: return nullptr;
    }
}

void flush_buffer(Flow::Timestamp* buffer) {
    auto u8_ptr = dynamic_cast<Flow::Vector<u8>*>(buffer);
    if (u8_ptr != nullptr) {
        dynamic_cast<Flow::Vector<u8>*>(buffer)->clear();
    }
    auto s16_ptr = dynamic_cast<Flow::Vector<s16>*>(buffer);
    if (s16_ptr != nullptr) {
        dynamic_cast<Flow::Vector<s16>*>(buffer)->clear();
    }
    auto s32_ptr = dynamic_cast<Flow::Vector<s32>*>(buffer);
    if (s32_ptr != nullptr) {
        dynamic_cast<Flow::Vector<s32>*>(buffer)->clear();
    }
    auto f32_ptr = dynamic_cast<Flow::Vector<f32>*>(buffer);
    if (f32_ptr != nullptr) {
        dynamic_cast<Flow::Vector<f32>*>(buffer)->clear();
    }
    auto f64_ptr = dynamic_cast<Flow::Vector<f64>*>(buffer);
    if (f64_ptr != nullptr) {
        dynamic_cast<Flow::Vector<f64>*>(buffer)->clear();
    }
}

}  // namespace

namespace Audio {

Core::ParameterInt FfmpegInputNode::paramResampleRate("resample-rate", "if > 0 audio input will be resampled to this sample rate", 0, 0);

std::once_flag FfmpegInputNode::ffmpeg_initialized;

struct FfmpegInputNode::Internal {
    AVFormatContext* fmt_ctx    = nullptr;
    AVCodecContext*  cdc_ctx    = nullptr;
    int              stream_idx = -1;
    SwrContext*      swr_ctx    = nullptr;
};

FfmpegInputNode::FfmpegInputNode(const Core::Configuration& c)
        : Core::Component(c), Node(c), Precursor(c), internal_(new Internal()), buffer_(nullptr), resampleRate_(paramResampleRate(c)) {
    std::call_once(FfmpegInputNode::ffmpeg_initialized, av_register_all);
}

bool FfmpegInputNode::setParameter(const std::string& name, const std::string& value) {
    if (paramResampleRate.match(name)) {
        resampleRate_ = paramResampleRate(value);
    }
    else {
        return Precursor::setParameter(name, value);
    }
    return true;
}

bool FfmpegInputNode::openFile_() {
    bool           success        = true;
    AVCodec*       codec          = nullptr;
    AVDictionary*  opts           = nullptr;
    s64            channel_layout = 0;
    AVSampleFormat input_fmt      = AV_SAMPLE_FMT_NONE;
    AVSampleFormat packed_fmt     = AV_SAMPLE_FMT_NONE;
    u32            output_sr      = 0;

    int error_code = avformat_open_input(&internal_->fmt_ctx, filename_.c_str(), nullptr, nullptr);
    if (error_code < 0) {
        error("could not open source file: ") << filename_;
        success = false;
        goto cleanup;
    }
    error_code = avformat_find_stream_info(internal_->fmt_ctx, nullptr);
    if (error_code < 0) {
        error("could not find stream info for: ") << filename_;
        success = false;
        goto cleanup;
    }

    error_code = av_find_best_stream(internal_->fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (error_code == AVERROR_STREAM_NOT_FOUND) {
        error("could not find audio-stream");
        success = false;
        goto cleanup;
    }
    else if (error_code == AVERROR_DECODER_NOT_FOUND) {
        error("could not find decoder for audio-stream");
        success = false;
        goto cleanup;
    }
    else if (error_code < 0) {
        error("other error in av_find_best_stream");
        success = false;
        goto cleanup;
    }
    else {
        internal_->stream_idx = error_code;
    }
    internal_->cdc_ctx = internal_->fmt_ctx->streams[internal_->stream_idx]->codec;

    if (internal_->cdc_ctx->channels == 1) {
	// explicitly ask for mono-layout, as swresample can throw an error if the input channel layout is 0 (=default)
        internal_->cdc_ctx->channel_layout = AV_CH_LAYOUT_MONO;
        internal_->cdc_ctx->request_channel_layout = AV_CH_LAYOUT_MONO;
    }
    else if (internal_->cdc_ctx->channels == 2) {
	// explicitly ask for stereo-layout, as swresample can throw an error if the input channel layout is 0 (=default)
        internal_->cdc_ctx->channel_layout = AV_CH_LAYOUT_STEREO;
        internal_->cdc_ctx->request_channel_layout = AV_CH_LAYOUT_STEREO;
    }

    av_dict_set(&opts, "refcounted_frames", "0", 0);
    error_code = avcodec_open2(internal_->cdc_ctx, codec, &opts);
    if (error_code < 0) {
        error("failed to open codec");
        success = false;
        goto cleanup;
    }

    channel_layout = internal_->cdc_ctx->channel_layout;
    input_fmt      = internal_->cdc_ctx->sample_fmt;
    packed_fmt     = av_get_packed_sample_fmt(input_fmt);
    output_sr      = resampleRate_ > 0 ? resampleRate_ : internal_->cdc_ctx->sample_rate;

    if (packed_fmt == AV_SAMPLE_FMT_NONE) {
        error("No suitable packed format");
        success = false;
        goto cleanup;
    }

    if (input_fmt != packed_fmt or resampleRate_ > 0) {
        internal_->swr_ctx = swr_alloc_set_opts(internal_->swr_ctx,
                                                channel_layout, packed_fmt, output_sr,
                                                channel_layout, input_fmt, internal_->cdc_ctx->sample_rate,
                                                0, nullptr);
        if (internal_->swr_ctx == nullptr) {
            error("could not allocate SwrContext");
            success = false;
            goto cleanup;
        }

	error_code = swr_init(internal_->swr_ctx);
	if (error_code) {
            error("could not initialize SwrContext: ") << error_code;
            success = false;
            goto cleanup;
	}
    }

    setSampleRate(output_sr);
    setSampleSize(8 * av_get_bytes_per_sample(packed_fmt));
    setTrackCount(internal_->cdc_ctx->channels);

    //set seeking cursor
    lastSeekTime_ = 0;

cleanup:
    av_dict_free(&opts);  // does this have to be done always or only on error?
    if (not success) {
        internal_->stream_idx = -1;
        if (internal_->swr_ctx != nullptr) {
            swr_free(&internal_->swr_ctx);
        }
        if (internal_->cdc_ctx != nullptr) {
            avcodec_close(internal_->cdc_ctx);
            internal_->cdc_ctx = nullptr;
        }
        if (internal_->fmt_ctx != nullptr) {
            avformat_close_input(&internal_->fmt_ctx);
            internal_->fmt_ctx = nullptr;
        }
    }

    return success;
}

void FfmpegInputNode::closeFile_() {
    internal_->stream_idx = -1;
    if (internal_->swr_ctx != nullptr) {
        swr_free(&internal_->swr_ctx);
    }
    if (internal_->cdc_ctx != nullptr) {
        avcodec_close(internal_->cdc_ctx);
        internal_->cdc_ctx = nullptr;
    }
    if (internal_->fmt_ctx != nullptr) {
        avformat_close_input(&internal_->fmt_ctx);
        internal_->fmt_ctx = nullptr;
    }
}

bool FfmpegInputNode::isFileOpen() const {
    return internal_->cdc_ctx != nullptr;
}

bool FfmpegInputNode::seek(SampleCount newSamplePos) {
    AVRational sr_norm     = av_make_q(1, internal_->cdc_ctx->sample_rate);
    s64        stream_time = av_rescale_q(newSamplePos, sr_norm, internal_->fmt_ctx->streams[internal_->stream_idx]->time_base);
    int        error_code  = avformat_seek_file(internal_->fmt_ctx, internal_->stream_idx, INT64_MIN, stream_time, stream_time, 0);
    if (error_code < 0) {
        error("error while seeking");
        return false;
    }
    avcodec_flush_buffers(internal_->cdc_ctx);
    sampleCount_ = newSamplePos;

    if (internal_->swr_ctx != nullptr) {
        // flush resampling buffer
        u32 output_sr = resampleRate_ > 0 ? resampleRate_ : internal_->cdc_ctx->sample_rate;

        AVFrame* in_frame  = av_frame_alloc();
        AVFrame* out_frame = av_frame_alloc();

        av_frame_set_channel_layout(in_frame, av_get_default_channel_layout(internal_->cdc_ctx->channels));
        av_frame_set_sample_rate(in_frame, internal_->cdc_ctx->sample_rate);
        out_frame->format = internal_->cdc_ctx->sample_fmt;

        av_frame_set_channel_layout(out_frame, av_get_default_channel_layout(internal_->cdc_ctx->channels));
        av_frame_set_sample_rate(out_frame, output_sr);
        out_frame->format = av_get_packed_sample_fmt(internal_->cdc_ctx->sample_fmt);

        do {
            swr_convert_frame(internal_->swr_ctx, out_frame, in_frame);
        } while (out_frame->nb_samples > 0);
        av_frame_free(&in_frame);
        av_frame_free(&out_frame);
    }

    // flush output buffer
    flush_buffer(buffer_);

    //set seek cursor
    lastSeekTime_ = stream_time;

    return true;
}

u32 FfmpegInputNode::read(u32 nSamples, Flow::Timestamp*& d) {
    AVPacket       packet;
    AVFrame*       frame       = nullptr;
    AVFrame*       out_frame   = nullptr;
    int            got_frame   = 0;
    size_t         buffer_size = 0ul;
    AVSampleFormat fmt         = AV_SAMPLE_FMT_NONE;

    av_init_packet(&packet);
    packet.data = nullptr;
    packet.size = 0;

    while (av_read_frame(internal_->fmt_ctx, &packet) >= 0) {
        if (packet.stream_index != internal_->stream_idx) {
            continue;
        }
        AVPacket orig_pkt = packet;
        frame             = av_frame_alloc();
        do {
            int ret = avcodec_decode_audio4(internal_->cdc_ctx, frame, &got_frame, &packet);
            if (ret < 0) {
                error("Error decoding frame");
                return 0u;
            }
            ret = FFMIN(packet.size, ret);

            if (got_frame) {
                if (internal_->swr_ctx != nullptr) {
                    out_frame = av_frame_alloc();
                    av_frame_set_channel_layout(out_frame, av_frame_get_channel_layout(frame));
                    av_frame_set_sample_rate(out_frame, resampleRate_ > 0 ? resampleRate_ : av_frame_get_sample_rate(frame));
                    out_frame->format = av_get_packed_sample_fmt(static_cast<AVSampleFormat>(frame->format));
                    int error_code    = swr_convert_frame(internal_->swr_ctx, out_frame, frame);
                    if (error_code < 0) {
                        error("Error converting frame: ");
                        return 0u;
                    }
                }
                else {
                    out_frame         = frame;
                    out_frame->format = av_get_packed_sample_fmt(static_cast<AVSampleFormat>(frame->format));
                }
                fmt = static_cast<AVSampleFormat>(out_frame->format);
                if (buffer_ == nullptr) {
                    buffer_ = alloc_buffer(fmt);
                    require(buffer_ != nullptr);
                }

                s64        time          = av_frame_get_best_effort_timestamp(out_frame);
                s64        time_offset   = lastSeekTime_ - time;
                AVRational sr_norm       = av_make_q(1, av_frame_get_sample_rate(out_frame));
                u32        sample_offset = static_cast<u32>(std::max(0l, av_rescale_q(time_offset, internal_->fmt_ctx->streams[internal_->stream_idx]->time_base, sr_norm)));
                buffer_size              = add_to_buffer(buffer_, out_frame, fmt, sample_offset);
            }
            if (out_frame != frame) {
                av_frame_free(&out_frame);
            }
            packet.size -= ret;
            packet.data += ret;
        } while (packet.size > 0);
        av_frame_free(&frame);
        av_free_packet(&orig_pkt);

        if (buffer_size >= nSamples * trackCount_) {
            break;
        }
    }
    packet.size = 0;
    packet.data = nullptr;

    d = create_output_buffer(buffer_, nSamples * trackCount_, fmt);
    return std::min<u32>(nSamples, buffer_size / trackCount_);
}

}  // namespace Audio
