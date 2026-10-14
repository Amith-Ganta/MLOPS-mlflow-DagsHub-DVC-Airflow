"""Pydantic models for the Product Return Prediction API."""

from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    """Raw order data — all fields a user would know pre-shipment."""

    product_category: str = Field(..., examples=["Electronics"])
    sub_category: str = Field(..., examples=["Smartphones"])
    brand: str = Field(..., examples=["BrandA"])
    product_price: float = Field(..., gt=0, examples=[499.99])
    discount_percent: float = Field(..., ge=0, le=100, examples=[15.0])
    product_rating: float = Field(..., ge=0, le=5, examples=[4.2])
    review_count: float = Field(..., ge=0, examples=[120])
    fragile_item: int = Field(..., ge=0, le=1, examples=[0])
    warranty_available: int = Field(..., ge=0, le=1, examples=[1])
    product_return_rate: float = Field(..., ge=0, le=1, examples=[0.12])
    category_return_rate: float = Field(..., ge=0, le=1, examples=[0.18])
    brand_return_rate: float = Field(..., ge=0, le=1, examples=[0.10])
    defect_rate: float = Field(..., ge=0, le=1, examples=[0.03])
    seller_rating: float = Field(..., ge=0, le=5, examples=[4.5])
    seller_return_rate: float = Field(..., ge=0, le=1, examples=[0.08])
    fulfillment_type: str = Field(..., examples=["Marketplace Fulfilled"])
    payment_method: str = Field(..., examples=["Credit Card"])
    quantity: int = Field(..., ge=1, examples=[1])
    shipping_distance_km: float = Field(..., ge=0, examples=[250.0])
    delayed_delivery: int = Field(..., ge=0, le=1, examples=[0])
    wishlist_before_purchase: int = Field(..., ge=0, le=1, examples=[1])
    product_page_views: int = Field(..., ge=0, examples=[35])
    customer_support_calls: int = Field(..., ge=0, examples=[1])
    chat_interactions: int = Field(..., ge=0, examples=[0])


class PredictionResponse(BaseModel):
    """Prediction result for a single order."""

    return_probability: float = Field(..., description="Probability the order will be returned (0-1)")
    predicted_returned: bool = Field(..., description="Whether the model predicts a return")
    risk_level: str = Field(..., description="Low / Medium / High risk label")


class MetricItem(BaseModel):
    name: str
    value: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelInfoResponse(BaseModel):
    """Information about the loaded model."""

    model_name: str
    metrics: list[MetricItem]
    feature_importance: list[FeatureImportanceItem]
    threshold: float
