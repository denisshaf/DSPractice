from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2


def write_schema(schema, output_path):
    schema_text = text_format.MessageToString(schema)
    file_io.write_string_to_file(output_path, schema_text)


def load_schema(input_path):
    schema = schema_pb2.Schema()
    schema_text = file_io.read_file_to_string(input_path)
    text_format.Parse(schema_text, schema)
    return schema
