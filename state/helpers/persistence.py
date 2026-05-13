"""bridges State and SessionManager for cross-turn persistence"""
import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from optorch.state import BaseState
    from optorch.session.session_manager import SessionManager


class PersistenceHelper:
    """round-trips state keys through session backend so they survive turns
    
    pydantic models marked with __model__ for clean rehydration
    """
    
    BAG_KEY = "__persisted_state__"
    
    @staticmethod
    def dump(value: Any) -> Any:
        """serialise pydantic with class marker"""
        if hasattr(value, "model_dump") and hasattr(value, "__class__"):
            cls = value.__class__
            return {
                "__model__": f"{cls.__module__}.{cls.__qualname__}",
                "data": value.model_dump(),
            }
        return value
    
    @staticmethod
    def load(value: Any) -> Any:
        """rehydrate pydantic from marker dump"""
        if isinstance(value, dict) and "__model__" in value and "data" in value:
            module_path, class_name = value["__model__"].rsplit(".", 1)
            try:
                cls = getattr(importlib.import_module(module_path), class_name)
                return cls(**value["data"])
            except Exception:
                return value["data"]
        return value
    
    @classmethod
    async def save(cls, sessions: "SessionManager", key: str, value: Any) -> None:
        """write a single key into the persisted bag"""
        if sessions.get_id() is None:
            return
        data = await sessions.get_data() or {}
        bag = dict(data.get(cls.BAG_KEY) or {})
        bag[key] = cls.dump(value)
        data[cls.BAG_KEY] = bag
        await sessions.set_data(data)
    
    @classmethod
    async def hydrate(cls, state: "BaseState", sessions: "SessionManager | None") -> None:
        """pull persisted bag into state — call at turn entry"""
        if sessions is None or sessions.get_id() is None:
            return
        data = await sessions.get_data() or {}
        bag = data.get(cls.BAG_KEY) or {}
        for key, raw in bag.items():
            if state.get(key) is not None:
                continue
            state.set(key, cls.load(raw))
