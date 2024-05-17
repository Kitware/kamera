# Sprokit Adapters
================

This package provides interfaces between ROS and Sprokit pipelines.

## kw_detector_adapter
===================

The `kw_detector_adapter_node` node provides an interface between ROS image
messages and a Sprokit detection pipeline. An example usage via a launch file
is provided in launch/sprokit_detector_adapter.launch. This launch file provides
additional documentation for all the ROS parameters that may be set to control
the node. Launch files for specific cases generally include this launch file,
passing the appropriate arguments.

An arbitrary number of image topics can be provided by enumerating ROS-topic
arguments 'image_in1', 'image_in2', etc., and images from each topic are then
multiplexed through the pipeline. The image topics may publish sensor_msgs Image
or custom_msg ImageWithMask messages, but the type must be specified with the
associated ROS arguments 'image_in1_type', 'image_in2_type', etc.

Each image received, from any of the source topics, is sent through the Sprokit
pipeline. The processing pipeline is defined by a pipe file, the path to which
is set via the `pipe_file` parameter. Example processing pipelines can be found
within the `pipelines` folder.

The kw_detector_adapter_node provides `input_adapter` and `output_adapter`
processes, and their associated ports must be connected within the pipe file.
`input_adapter` provides ports `image` and `header` with the additional port
`mask` provided when image type is `ImageWithMask`. `output_adapter` requires
connections to ports `raw_image`, `header`, `detected_object_set`, and
'annotated_image'. `raw_image` and `header` are generally passed from the
`input_adapter` ports `image` and `header`.
