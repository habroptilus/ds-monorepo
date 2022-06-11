# House price

## eda
* 排反なカテゴリはorderedだけ
* city_codeとcityは一対一
* layoutの+Sはサービスルームらしい
  
## 効いたもの/効かなかったもの
効いたもの
* nearest_staのtarget_encode
* districtをtarget_encode
* cityごとにbuilt_year_seireki,areaの集約
* districtごとにbuilt_year_seireki,areaの集約
* lda(city,layout,5) (微改善)
* nearest_staごとにbuilt_year_seireki,areaの集約 (微改善)


改善しなかったもの
* layoutごとのareaの集約 -> これほんと？再確認する
* city_l1,layout_l1をtarget_encode
* cityとlayoutを組み合わせてtarget encode
* districtをlayoutでldaベクトル化
* cityのtarget_encode


## アイデア

* [x] cityを市と区にわける
* [x] build yearを西暦に変える
* [x] 築何年を計算する
* [x] +Sを分離したものを付け加える
* [x] orderedの年と半期を分離する
* [x] layout_l1ごとにareaの平均.diffを取る -> あんまり改善せず
* [x] city codeを削除する
* [x] structureをmultihotに開く