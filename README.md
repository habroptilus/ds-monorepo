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

## Development

パッケージの追加は以下の通り.

`poetry add (-D) パッケージ名`

開発用のパッケージはオプションをつける。

仮想環境に入る場合は`poetry shell`



