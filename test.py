from src.Tools.LibRASR import DataSource
from src.Tools.LibRASR import AbstractNode
from src.Tools.LibRASR import InputNode

# node_name
# source_
# sample_rate
# sample_type
# track_count

##########

node = source_.get_node(node_name);

if isinstance(node, InputNode):
    input_node_ = node
    print("Successfully casted to InputNode")
else:
    print("Failed to cast to InputNode")

# Set inputNode_
input_node_.set_parameter("sample-rate", sample_rate)
input_node_.set_parameter("sample-type", sample_type)
input_node_.set_parameter("track-count", track_count)
# input_node_.set_byte_stream_appender(std::bind(&FlowProcessor::appendDataToInputNode, this, std::placeholders::_1)); 
# Instead of appendDataToInputNode, pass a python function that append all samples from wav file to the sample_queue

# Set source_
source_.configure()
source_.reset()
# source_.configure_all() # doesn't exist!

# Do feature extraction
# while (data.get() != Flow::Data::eos()) { source_.getData(portIds_[p], data); // dump data in file }
