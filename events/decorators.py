"""auto-emits start/complete/error events with timing"""

from typing import Callable, Any, TypeVar, overload, Awaitable, AsyncIterator, TYPE_CHECKING
import time
import asyncio
from functools import wraps
from optorch.decorators.context_extraction import extract_context
from optorch.state import State
from optorch.utils.json_encoder import make_json_safe

if TYPE_CHECKING:
    from optorch.llm.lifecycle.context import LLMContext

T = TypeVar('T')


def _lookup_session_id(llm_context:'LLMContext') -> str | None:
    """extract session_id from llm_context if available"""
    session_id = None
    if hasattr(llm_context, 'state') and llm_context.state:
        session_id = llm_context.state.get("session_id") if hasattr(llm_context.state, 'get') else None
    if not session_id and hasattr(llm_context, 'metadata') and llm_context.metadata:
        session_id = llm_context.metadata.get("session_id")
    
    return session_id


def _lookup_model(args: tuple) -> str | None:
    """pull model from bound instance .model or from an LLMContext arg"""
    if not args:
        return None
    model = getattr(args[0], 'model', None)
    if model:
        return model
    for a in args:
        cfg = getattr(a, 'config', None)
        if isinstance(cfg, dict):
            m = cfg.get('model')
            if m:
                return m
    return None


def _resolve_session_id(context: Any, llm_context: Any) -> str | None:
    """llm_context > context.state > ambient ContextVar"""
    if llm_context:
        sid = _lookup_session_id(llm_context)
        if sid:
            return sid
        
    state = getattr(context, 'state', None) if context is not None else None
    
    if state is not None and hasattr(state, 'get'):
        sid = state.get('session_id')
        if sid:
            return sid
    
    from optorch.session.session_manager import _current_session
    return _current_session.get()


@overload
def emits(event_prefix: str) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]: ...

@overload
def emits(event_prefix: str) -> Callable[[Callable[..., AsyncIterator[T]]], Callable[..., AsyncIterator[T]]]: ...

@overload
def emits(event_prefix: str) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


def emits(event_prefix: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to auto-emit start/complete/error events.
    
    Extracts NodeContext from args/kwargs to access EventEmitter.
    If context not found, silently skips emission (useful for testing).
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def filter_kwargs(kwargs: dict) -> dict:
            return {
                k: v for k, v in kwargs.items() 
                if not isinstance(v, State) 
                and k != 'context' 
                and not (hasattr(v, '__class__') and 'Context' in v.__class__.__name__)
            }
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = extract_context(args, kwargs)
            
            llm_context: Any = None
            if context and hasattr(context, '__class__') and 'LLMContext' in context.__class__.__name__:
                llm_context = context
            else:
                for arg in args:
                    if hasattr(arg, '__class__') and 'LLMContext' in arg.__class__.__name__:
                        llm_context = arg
                        break

            session_id = _resolve_session_id(context, llm_context)

            node_name = context.current_node_name if context and hasattr(context, 'current_node_name') else None
            serializable_kwargs = filter_kwargs(kwargs)
            model = _lookup_model(args)
            start_time = time.time()
            
            event_data: dict[str, Any] = {"args": serializable_kwargs}
            if node_name:
                event_data["node_name"] = node_name
            if session_id:
                event_data["session_id"] = session_id
            if model:
                event_data["model"] = model
            
            if context and hasattr(context, 'events'):
                context.events.emit(f"{event_prefix}.start", event_data)
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)
                
                complete_data: dict[str, Any] = {
                    "duration_ms": duration_ms,
                    "result": make_json_safe(result),
                    "args": serializable_kwargs
                }

                if node_name:
                    complete_data["node_name"] = node_name

                if session_id:
                    complete_data["session_id"] = session_id

                if model:
                    complete_data["model"] = model
                
                if llm_context and hasattr(llm_context, 'metadata') and llm_context.metadata:
                    complete_data.update({k: v for k, v in llm_context.metadata.items() if k not in complete_data})
                
                if context and hasattr(context, 'events'):
                    context.events.emit(f"{event_prefix}.complete", complete_data)
                
                return result
            except Exception as e:
                error_data: dict[str, Any] = {"error": str(e)}
                if node_name:
                    error_data["node_name"] = node_name

                if session_id:
                    error_data["session_id"] = session_id

                if model:
                    error_data["model"] = model

                if context and hasattr(context, 'events'):
                    context.events.emit(f"{event_prefix}.error", error_data)

                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = extract_context(args, kwargs)
            
            llm_context: Any = None
            if context and hasattr(context, '__class__') and 'LLMContext' in context.__class__.__name__:
                llm_context = context
            else:
                for arg in args:
                    if hasattr(arg, '__class__') and 'LLMContext' in arg.__class__.__name__:
                        llm_context = arg
                        break

            session_id = _resolve_session_id(context, llm_context)

            node_name = context.current_node_name if context and hasattr(context, 'current_node_name') else None
            serializable_kwargs = filter_kwargs(kwargs)
            model = _lookup_model(args)
            start_time = time.time()
            
            event_data: dict[str, Any] = {"args": serializable_kwargs}
            if node_name:
                event_data["node_name"] = node_name
            if session_id:
                event_data["session_id"] = session_id
            if model:
                event_data["model"] = model
            
            if context and hasattr(context, 'events'):
                context.events.emit(f"{event_prefix}.start", event_data)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)
                
                complete_data: dict[str, Any] = {
                    "duration_ms": duration_ms,
                    "result": make_json_safe(result),
                    "args": serializable_kwargs
                }

                if node_name:
                    complete_data["node_name"] = node_name

                if session_id:
                    complete_data["session_id"] = session_id

                if model:
                    complete_data["model"] = model
                
                if llm_context and hasattr(llm_context, 'metadata') and llm_context.metadata:
                    complete_data.update({k: v for k, v in llm_context.metadata.items() if k not in complete_data})
                
                if context and hasattr(context, 'events'):
                    context.events.emit(f"{event_prefix}.complete", complete_data)
                
                return result
            except Exception as e:
                error_data: dict[str, Any] = {"error": str(e)}
                if node_name:
                    error_data["node_name"] = node_name

                if session_id:
                    error_data["session_id"] = session_id

                if model:
                    error_data["model"] = model

                if context and hasattr(context, 'events'):
                    context.events.emit(f"{event_prefix}.error", error_data)

                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
