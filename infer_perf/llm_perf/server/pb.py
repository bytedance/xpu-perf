from typing import Any, Generator, Iterable, List

from llm_perf.server import server_pb2, server_pb2_grpc


def deserialize_value(value: server_pb2.Value) -> Any:
    kind = value.WhichOneof("kind")
    if kind == "float_":
        return value.float_
    elif kind == "int64_":
        return value.int64_
    elif kind == "bytes_":
        return value.bytes_
    elif kind == "string_":
        return value.string_
    elif kind == "float_list":
        return [v for v in value.float_list.values]
    elif kind == "int64_list":
        return [v for v in value.int64_list.values]
    elif kind == "bytes_list":
        return [v for v in value.bytes_list.values]
    elif kind == "string_list":
        return [v for v in value.string_list.values]
    elif kind == "struct_":
        return {k: deserialize_value(v) for k, v in value.struct_.fields.items()}
    else:
        raise TypeError(f"Invalid type {type(value)}")


def serialize_value(value: Any) -> server_pb2.Value:
    if isinstance(value, float):
        return server_pb2.Value(float_=value)
    elif isinstance(value, int):
        return server_pb2.Value(int64_=value)
    elif isinstance(value, bytes):
        return server_pb2.Value(bytes_=value)
    elif isinstance(value, str):
        return server_pb2.Value(string_=value)
    elif isinstance(value, list):
        if isinstance(value[0], float):
            return server_pb2.Value(float_list=server_pb2.FloatList(values=value))
        elif isinstance(value[0], int):
            return server_pb2.Value(int64_list=server_pb2.Int64List(values=value))
        elif isinstance(value[0], bytes):
            return server_pb2.Value(bytes_list=server_pb2.BytesList(values=value))
        elif isinstance(value[0], str):
            return server_pb2.Value(string_list=server_pb2.StringList(values=value))
    elif isinstance(value, dict):
        return server_pb2.Value(
            struct_=server_pb2.Struct(
                fields={k: serialize_value(v) for k, v in value.items()}
            )
        )
    else:
        raise TypeError(f"Invalid type {type(value)}")