"""Factoryクラスのベースクラス."""
from inspect import signature
from abc import ABCMeta


class FactoryBase(metaclass=ABCMeta):
    """Factoryクラスの基底クラス."""

    def __init__(self, str2model, custom_members, shared_params=None):
        """str2modelを引数にすることを強制する.

        :params
        str2model: モデルフラグ->モデルのmapping(dict)
        custom_members: 追加でカスタムモデルを追加登録する(dictから一括で登録可能)
        shared_params: どのモデルに与えるパラメータにも共通で渡したいものはこちらにセットする.
        """
        self.str2model = str2model
        self.shared_params = {} if shared_params is None else shared_params
        if custom_members:
            self.register_models_from_dict(custom_members)

    def register_model(self, model_str, Model):
        """モデルを登録する."""
        if model_str in self.str2model:
            print(
                f"[WARNING] You are overwriting '{model_str}' with {Model}.")
        self.str2model[model_str] = Model

    def register_models_from_dict(self, custom_members):
        """モデルフラグをkey、モデルvalueとするdictを受け取り一括でモデルを登録する."""
        for key, model in custom_members.items():
            self.register_model(key, model)

    def get_model(self, model_str):
        """指定されたモデルフラグに対応するモデルを返す.なければ例外を投げて終了する."""
        if model_str in self.str2model:
            return self.str2model[model_str]
        else:
            raise Exception(f"Invalid model flag '{model_str}'.")

    def get_required_params_list(self, Model):
        """Modelのinitに必要なパラメータのリストを返す."""
        required_params_list = list(signature(
            Model.__init__).parameters.keys())
        required_params_list.remove("self")
        return required_params_list

    def update_shared_params(self, params):
        """shared_paramsをベースに、paramsで更新する."""
        shared = self.shared_params.copy()
        shared.update(params)
        return shared

    def run(self, model_str, params=None):
        """モデルフラグを受け取り、必要な引数を取り出してモデルに与えてインスタンスを作成して返す."""
        params = {} if params is None else params
        Model = self.get_model(model_str)
        required_params_names = self.get_required_params_list(Model)
        all_params = self.update_shared_params(params)
        required_params = self.extract_required_params(
            required_params_names, all_params)
        return Model(**required_params)

    def extract_required_params(self, required_params_names, all_params):
        """必要な引数を取り出す. 必要な引数が与えられていない場合にWARNINGを出す."""
        required_params = {}
        for params_name in required_params_names:
            if params_name not in all_params:
                print(
                    f"[WARNING] parameter '{params_name}' is not specified. So default will be used.")
                continue
            required_params[params_name] = all_params[params_name]
        return required_params
