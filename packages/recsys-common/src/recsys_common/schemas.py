from pydantic import BaseModel, Field


class RawUserFeatures(BaseModel):
    user_id: str
    user_age: float
    user_historic_categories: list[str] = Field(default_factory=list)
    session_last_3_clicks: list[str] = Field(default_factory=list)


class RawItemFeatures(BaseModel):
    item_id: str
    item_category: str
    item_historical_ctr: float
    item_price: float


class ContextFeatures(BaseModel):
    device_type: str
    hour_of_day: int
