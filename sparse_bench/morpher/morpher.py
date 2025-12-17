from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Callable, Type
import torch.nn as nn
from abc import ABC, abstractmethod
import types


@dataclass
class AttributeReplacement:
    name: str
    value: Any

@dataclass
class MethodReplacement:
    name: str
    target_method: Callable


@dataclass
class ModuleTransformation:
    attribute_replacement: List[AttributeReplacement] = field(default_factory=list)
    method_replacement: List[MethodReplacement] = field(default_factory=list)



class Strategy(ABC):

    def __init__(self):
        self.model = None

    def preprocess(self, model: nn.Module):
        return model
    

    def postprocess(self, model: nn.Module):
        return model

    @abstractmethod
    def get_module_transformation(self) -> Dict[Union[str, Type[nn.Module]], ModuleTransformation]:
        pass


class Morpher:
    def __init__(self, model: nn.Module, strategy: Strategy):
        self.model = model
        self.strategy = strategy

    def transform(self) -> nn.Module:
        model = self.strategy.preprocess(self.model)
        module_transformations = self.strategy.get_module_transformation()
        print(f"applying transformation: {module_transformations}")

        for target, transformation in module_transformations.items():
            target: Union[str, Type[nn.Module]]
            if target == "":
                target_module = [model]
            elif isinstance(target, str):
                target_module = [model.get_submodule(target)]
            else:
                # find all modules of the type target
                target_module = [m for m in model.modules() if isinstance(m, target)]

            if len(target_module) == 0:
                raise ValueError(f"Module {target} not found in model")

            for module in target_module:
                module = self._apply_single_transformation(module, transformation)

        model = self.strategy.postprocess(model)
        return model


    def _apply_single_transformation(self, module: nn.Module, transformation: ModuleTransformation):
        for attribute_replacement in transformation.attribute_replacement:
            self._update_attribute(module, attribute_replacement.name, attribute_replacement.value)
        for method_replacement in transformation.method_replacement:
            self._update_method(module, method_replacement.name, method_replacement.target_method)
        return module

    def _update_method(self, module: nn.Module, name: str, target_method: Callable):
        if hasattr(module, name):
            # bind the method to the object
            setattr(module, name, types.MethodType(target_method, module))
        else:
            raise ValueError(f"Module {module} does not have method {name}")

    def _update_attribute(self, module: nn.Module, name: str, value: Any):
        
        if hasattr(module, name):
            setattr(module, name, value)
        else:
            raise ValueError(f"Module {module} does not have attribute {name}")

