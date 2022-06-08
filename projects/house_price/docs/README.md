# House price

## eda
* 排反なカテゴリはorderedだけ
* city_codeとcityは一対一
* layoutの+Sはサービスルームらしい
  

## アイデア

* [x] cityを市と区にわける
* [x] build yearを西暦に変える
* [x] 築何年を計算する
* [x] +Sを分離したものを付け加える
* [x] orderedの年と半期を分離する
* [ ] layoutごとにareaの平均.diffを取る
* [x] city codeを削除する
* [x] structureをmultihotに開く