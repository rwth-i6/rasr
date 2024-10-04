from src.Tools.LibRASR import Configuration
from src.Tools.LibRASR import DataSource
from src.Tools.LibRASR import AbstractNode
from src.Tools.LibRASR import InputNode
from src.Tools.LibRASR import Data
import wave

flow_config = Configuration()
flow_config.set_from_file("feature.config")
source_ = DataSource(flow_config, True)

node_name = "audio"
input_node_ = source_.get_node(node_name);

with wave.open("8288-274162-0066.wav", "rb") as wav_file:
    sample_rate = str(wav_file.getframerate())
    sample_width = wav_file.getsampwidth() # double check values
    if sample_width == 1:
        sample_type = "u8" 
    elif sample_width == 2:
        sample_type = "s16"
    elif sample_width == 4:
        sample_type = "f32"
    else:
        raise ValueError("Unsupported sample width")
    track_count = str(wav_file.getnchannels())

# Set input_node_
input_node_.set_parameter("sample-rate", sample_rate)
input_node_.set_parameter("sample-type", sample_type)
input_node_.set_parameter("track-count", track_count)

def append_data_to_input_node(sample_queue):
    with wave.open("8288-274162-0066.wav", "rb") as wav_file:
        wave_char = wav_file.readframes(wav_file.getnframes())

    sample_queue = [w for w in wave_char]

input_node_.set_byte_stream_appender(append_data_to_input_node) # unsure 

# Set source_
source_.configure()
source_.reset()

# Do feature extraction
data = Data()

while (data.compare_address(Data.eos())):
    source_.get_data(0, data); # dump data in file 
