from src.Tools.LibRASR import Configuration
from src.Tools.LibRASR import DataSource
from src.Tools.LibRASR import AbstractNode
from src.Tools.LibRASR import InputNode
import wave

# sample_queue // dequeue/list of samples of wav //
# data         // Flow::DataPtr<Flow::Data>      //

##########

flow_config = Configuration()
flow_config.set_from_file("feature.config") # BUG
source_ = DataSource(flow_config, True)

node_name = "samples"
input_node_ = source_.get_node(node_name);

print(type(input_node_)) # BUG

with wave.open("8288-274162-0066.wav", "rb") as wav_file:
    sample_rate = str(wav_file.getframerate())
    sample_width = wav_file.getsampwidth()
    if sample_width == 1:
        sample_type = "SampleTypeU8"
    elif sample_width == 2:
        sample_type = "SampleTypeS16"
    elif sample_width == 4:
        sample_type = "SampleTypeF32"
    else:
        raise ValueError("Unsupported sample width")
    track_count = str(wav_file.getnchannels())

# Set input_node_
input_node_.set_parameter("sample-rate", sample_rate)
input_node_.set_parameter("sample-type", sample_type)
input_node_.set_parameter("track-count", track_count)
