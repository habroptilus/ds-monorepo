# ds-monorepo

## How to setup

```
python -V
3.8.11

poetry -V
Poetry version 1.1.12
```

パッケージのインストールを行う.

`poetry install`

その際、ml_metricsのinstallでエラーが出るかもしれない。
その時は以下のコマンドを実行した後、再度`poetry install`を実行すること。

`poetry run pip install -U "setuptools<58"`
`poetry run pip install xfeat bhtsne`

See https://github.com/pfnet-research/xfeat/issues/9.


(ここ一発でいけるようにしたい)


次にlilacのパスを通す。

lilacが上位ディレクトリにあるため、各projectからimportできないので、環境変数`PYTHONPATH`にあらかじめlilacまでのパスを入れておく。

.bashrcなどに以下を書いておくなどする.

```
export PYTHONPATH="<ds-monorepoまでのフルパス>:$PYTHONPATH"
```

(ここもどうにかしたい)

## lilac commands

`lilac`コマンドをいくつか提供している。

### Run experiment

`poetry run lilac run -p <プロジェクト名> -f <実験設定のyamlファイル名> [-s] `
で実験が実行され、CVが表示され、デフォルトだと`projects/<プロジェクト名>/data/output/<実験設定のyamlファイル名.json>`
に結果ファイルが出力される.
`-s` をつけるとモデルが保存される.
### List results

`poetry run lilac list -p <プロジェクト名> -n <直近n回の実験結果を見る> `
でCVのリストを表示する.

### Show detail

`poetry run lilac detail -p <プロジェクト名> -e <実験番号(001など)>`

で指定した実験の詳細結果を表示する。

### Plot importance

`poetry run lilac plot -p <プロジェクト名> -e <実験番号(001など)> -n <plotする特徴量上位いくつか>`

で、fold1のmodelがfeature_importanceを計算していれば上位n個のカラムを出力する.
## Development

パッケージの追加は以下の通り.

`poetry add (-D) パッケージ名`

開発用のパッケージはオプションをつける。

仮想環境に入る場合は`poetry shell`


## Lint and Format

```
make lint
make format
```

でそれぞれ実行できる. `lilac/`と`projects/`以下のコードが対象.


## Pitfalls

* diffモデルでndarrayに変換しないとおかしなことになる
* trainデータ数を変えた実験をロードしてアンサンブルするとこけるのに注意
