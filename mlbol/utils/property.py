import inspect
from typing import Callable
from typing import Any

__all__ = [
    "ClassPropertyDescriptor",
    "ClassPropertyMetaClass",
    "classproperty",
    "lazyproperty",
]


def lazyproperty(fn: Callable) -> property:
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazyproperty(self) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyproperty


class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        if inspect.isclass(obj):
            type_ = obj
            obj = None
        else:
            type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func: Callable) -> "ClassPropertyDescriptor":
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


class ClassPropertyMetaClass(type):
    def __new__(metacls, name, bases, namespace, **kwargs):
        for base in bases:
            for k, v in base.__dict__.items():
                if k not in namespace and type(v) is ClassPropertyDescriptor:
                    namespace.update({k: v})
        return super().__new__(metacls, name, bases, namespace, **kwargs)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            obj = self.__dict__.get(key)
            if obj and type(obj) is ClassPropertyDescriptor:
                return obj.__set__(self, value)
        return super(ClassPropertyMetaClass, self).__setattr__(key, value)


def classproperty(func: Callable) -> ClassPropertyDescriptor:
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)
