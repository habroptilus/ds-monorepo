# ds-monorepo

## How to setup

Python3.8を想定している。

```
python -V
3.8.11
```
`poetry install`

ml_metricsのinstallでエラーが出るかもしれない。
その時は以下のコマンドを実行した後、再度`poetry install`を実行すること。

(ここ一発でいけるようにしたい)


次にlilacのパスを通す。

lilacが上位ディレクトリにあるため、各projectからimportできないので、環境変数`PYTHONPATH`にあらかじめlilacまでのパスを入れておく。

.bashrcなどに以下を書いておくなどする.

```
export PYTHONPATH="<ds-monorepoまでのフルパス>:$PYTHONPATH"
```

## lilac commands

`poetry run lilac`コマンドをいくつか提供している。

### Run experiment

`poetry run lilac run -p <プロジェクト名> -f <実験設定のyamlファイル名> `
で実験が実行され、CVが表示され、デフォルトだと`projects/<プロジェクト名>/data/output/<実験設定のyamlファイル名.json>`
に結果ファイルが出力される.

### List results

`poetry run lilac list -p <プロジェクト名> -n <直近n回の実験結果を見る> `
でCVのリストを表示する.

### Show detail

`poetry run lilac detail -p <プロジェクト名> -e <実験番号(001など)>`

で指定した実験の詳細結果を表示する。

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
