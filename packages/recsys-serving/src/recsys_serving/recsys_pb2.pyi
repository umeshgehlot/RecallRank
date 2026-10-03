from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class RecommendRequest(_message.Message):
    __slots__ = ("user_id", "device_type", "hour_of_day", "top_n", "retrieval_k")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    HOUR_OF_DAY_FIELD_NUMBER: _ClassVar[int]
    TOP_N_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_K_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    device_type: str
    hour_of_day: int
    top_n: int
    retrieval_k: int
    def __init__(self, user_id: str | None = ..., device_type: str | None = ..., hour_of_day: int | None = ..., top_n: int | None = ..., retrieval_k: int | None = ...) -> None: ...

class RecommendResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[RecommendedItem]
    def __init__(self, items: _Iterable[RecommendedItem | _Mapping] | None = ...) -> None: ...

class RecommendedItem(_message.Message):
    __slots__ = ("item_id", "score", "rank", "ctr_score", "cvr_score")
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    CTR_SCORE_FIELD_NUMBER: _ClassVar[int]
    CVR_SCORE_FIELD_NUMBER: _ClassVar[int]
    item_id: str
    score: float
    rank: int
    ctr_score: float
    cvr_score: float
    def __init__(self, item_id: str | None = ..., score: float | None = ..., rank: int | None = ..., ctr_score: float | None = ..., cvr_score: float | None = ...) -> None: ...
