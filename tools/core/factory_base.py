"""Factoryクラスのベースクラス."""
from inspect import signature
from abc import ABCMeta, abstractmethod


class FactoryBase(metaclass=ABCMeta):
    """Factoryクラスの基底クラス."""

    def __init__(self, str2model, custom_members=None):
        self.str2model = str2model
        if custom_members:
            self.register_models_from_dict(custom_members)

    def register_model(self, model_str, Model):
        if model_str in self.str2model:
            print(
                f"[WARNING] You are overwriting '{model_str}' with {Model}.")
        self.str2model[model_str] = Model

    def register_models_from_dict(self, custom_members):
        for key, model in custom_members.items():
            self.register_model(key, model)

    def get_model(self, model_str):
        if model_str in self.str2model:
            return self.str2model[model_str]
        else:
            raise Exception(f"Invalid model flag {model_str}")

    def get_required_params_list(self, Model):
        required_params_list = list(signature(
            Model.__init__).parameters.keys())
        required_params_list.remove("self")
        return required_params_list

    @abstractmethod
    def get_params(self, *args, **kwargs):
        raise Exception("Please implement.")

    def run(self, model_str, *args, **kwargs):
        """必要な引数が与えられていない場合にWARNINGを出す."""
        Model = self.get_model(model_str)
        required_params_list = self.get_required_params_list(Model)
        params = self.get_params(*args, **kwargs)
        required_params = {}
        for params_name in required_params_list:
            if params_name not in params:
                print(
                    f"[WARNING] parameter '{params_name}' is not specified for '{model_str}'. So default will be used.")
                continue
            required_params[params_name] = params[params_name]
        return Model(**required_params)
