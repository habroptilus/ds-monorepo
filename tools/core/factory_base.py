"""Factoryクラスのベースクラス."""
from inspect import signature
from abc import ABCMeta
from importlib import import_module
import re


class FactoryBase(metaclass=ABCMeta):
    """Factoryクラスの基底クラス.

    以下の機能を提供する.
    * モデルフラグを受け取って引数を渡してモデルをインスタンス化して返す.
    * 必要のないパラメータを渡しても問題ないようにしている.(ここをエラーにするオプションをつけてもよいがtodo)
    * 必要なパラメータがない場合、かつモデル側がデフォルト引数を持っている場合、WARNINGを出す.
    継承して利用する場合は__init__をオーバーライドするだけ.
    """

    def __init__(self, str2model, register_from, shared_params=None):
        """str2modelを引数にすることを強制する.

        :params
        str2model: モデルフラグ->モデルのmapping(dict)
        custom_members: 追加でカスタムモデルを追加登録する(dictから一括で登録可能)
        shared_params: どのモデルに与えるパラメータにも共通で渡したいものはこちらにセットする.
        """
        self.str2model = str2model
        self.shared_params = {} if shared_params is None else shared_params
        if register_from:
            self.register_models_from_src(register_from)

    def register_model(self, model_str, Model):
        """一つのモデルを登録する."""
        if model_str in self.str2model:
            print(
                f"[WARNING] You are overwriting '{model_str}' with {Model}.")
        self.str2model[model_str] = Model

    def register_models_from_src(self, src):
        """複数モデルを一括で登録する.srcはファイルパスかdictかを選択できる.

        ファイルパス経由の場合、キーはクラス名をスネークケースにしたものが自動で設定される.
        """
        if type(src) is str:
            src = self.get_custom_members_from_filepath(src)
        elif type(src) is not dict:
            raise Exception("Invalid src for registration of factory.")

        self.register_models_from_dict(src)

    def get_custom_members_from_filepath(self, filepath):
        """filepathからBaseを継承したものを取ってきて、スネークケース:クラスのdictを返す"""
        # TODO: カスタムモデルのファイルが複数ある場合に対応する
        # filepathだけどcustom.feature_generatorsのようなものを想定
        a = import_module(filepath)

        flag = False
        result = {}
        for k, v in vars(a).items():
            if flag:
                camel_key = re.sub("([A-Z])", lambda x: "_" +
                                   x.group(1).lower(), k)[1:]
                result[camel_key] = v
            if "Base" in k:
                flag = True
        return result

    def register_models_from_dict(self, members_dict):
        """モデルフラグをkey、モデルvalueとするdictを受け取り一括でモデルを登録する."""
        for key, model in members_dict.items():
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
