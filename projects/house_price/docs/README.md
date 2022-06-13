# House price

## eda
* 排反なカテゴリはorderedだけ
* city_codeとcityは一対一
* layoutの+Sはサービスルームらしい
  
## 効いたもの/効かなかったもの
効いたもの：
* nearest_staのtarget_encode
* districtをtarget_encode
* cityごとにbuilt_year_seireki,areaの集約
* districtごとにbuilt_year_seireki,areaの集約
* lda(city,layout,5) (微改善)
* nearest_staごとにbuilt_year_seireki,areaの集約 (微改善)
* depthを5->8に変えた
* seed average (seedによるブレがあるのをなくすことに貢献していそう. 改善はしてない？)


改善しなかったもの：
* layoutごとのareaの集約
* city_l1,layout_l1をtarget_encode
* cityとlayoutを組み合わせてtarget encode
* districtをlayoutでldaベクトル化
* cityのtarget_encode
* nearest_staとbuilt_yearのconcatをtarget encode
* detphを変えて平均アンサンブル(3,5,8)
* area x floor_ratioで延べ床面積



## アイデア

* [x] cityを市と区にわける
* [x] build yearを西暦に変える
* [x] 築何年を計算する
* [x] +Sを分離したものを付け加える
* [x] orderedの年と半期を分離する
* [x] layout_l1ごとにareaの平均.diffを取る -> あんまり改善せず
* [x] city codeを削除する
* [x] structureをmultihotに開く
* [x] area x floor_ratioで延べ床面積っぽいものができそう
* [ ] 集約をmean以外も試す
* [ ] 集約をnearest_min, ordered_year,ageも試す
* [ ] catboostとlgbmをアンサンブルする
* [x] 部屋数
* [x] 一部屋当たりの面積
* [x] 何階立てか
* [ ] diff ratioをもう一段集約する

## 疑問

* layout,layout_l1を見るとかなり予測に効きそうなのにtarget_encodingで改善しないのはなぜ？   
  * ある程度ルールがわかるのでいい感じに順序を与えてあげるとか


## 参考

https://static.signate.jp/competitions/182/summaries/3%E4%BD%8D.pdf
