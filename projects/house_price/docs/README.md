# House price


## eda
* 排反なカテゴリはorderedだけ
* city_codeとcityは一対一
* layoutの+Sはサービスルームらしい
* 極端に小さいprice_logのものがあって、それが予測を外している
  * structureがRCやSRC, layoutが3LDKあたり,年代は昭和から平成初期にかけてのものに含まれる
  * ただ、それらは平均で見ると普通、むしろ高い
* 13210行trainにはid以外同じなのにprice_logが異なるものが存在する
  * 予測しようがないよねこれ
  * でもなぜかgroupbyすると三つしかなくなる
  * わかった、13210行の方はnullのものも含んでいる.
  * 三つはnullもなくて重複している
    * dup_id_list=["id_40088089", "id_40088088", "id_23116490", "id_23116486", "id_12027757","id_12027758"]
* 上の13210行とは別でprice_logまで重複しているのは7911行。v5で削除した  
## 効いたもの/効かなかったもの
効いたもの：
* nearest_staのtarget_encode
* districtをtarget_encode
* cityごとにbuilt_year_seireki,areaの集約
* districtごとにbuilt_year_seireki,areaの集約
* nearest_staごとにbuilt_year_seireki,areaの集約 (微改善)
* nearest_minを集約に追加(とんとん~微改善)
* lda(city,layout,5) (微改善)
* depthを5->8に変えた
* seed average (seedによるブレがあるのをなくすことに貢献していそう. 改善はしてない？)
* logとって集約(特に分布が正規分布に近づくものは有効？)
  * オリジナルの集約もどちらも改善に効いている
* diffモデル(targetをprice_log-area_logに変更したもの)
  * 特にアンサンブルで改善
* min_child_samplesを大きくする
  * 20->200で改善
  * 200->2000は悪化
* diff ratioをもう一段集約 -> 微改善
* colsample_bytreeを1->0.7で微微改善
  * 誤差レベルな気もする


改善しなかったもの：
* city_l1,layout_l1をtarget_encode
* cityのtarget_encode
* nearest_staとbuilt_yearのconcatをtarget encode
* cityとlayoutを組み合わせてtarget encode
* layoutごとのareaの集約
* districtをlayoutでldaベクトル化
* detphを変えて平均アンサンブル(3,5,8)
* area x floor_ratioで延べ床面積(悪化する)
* 延べ床面積、部屋数、一部屋あたりの面積、何回立てかを入れたv3 (悪化する...)
  * total_floor_areaだけ上位に来ていて、それが悪さしている？
  * 目的変数と散布図を書くと予測に効きやすそうな感じしているけどな。。。なぜ？
* v4のdistrict-built_yearのtarget encode
* v6
* 正規化してxentropyでtrainする(アンサンブルにはワンチャン使えるかも)
  * 平均アンサンブルでは改善しなかった
* v7(v3と同じ)はやっぱり悪化





## アイデア

* [x] catboostのdiff
* [x] logとったものでもう一度diffratioの2段集約
* [x] 正規化してxentropyでtrainする
* [x] seed averaging
* [x] min_child_samplesを大きくする
* [x] trainの重複を削除する
* [x] 集約に用いたnearest_min,areaをlogに変換してから集約する
  * 正規分布っぽくなるからよいかも？
* [x] 集約をnearest_min, age,試す
* [x] v4のdistrict-built_yearのtarget encode
* [x] RMSEとMAEのアンサンブル
* [x] cityを市と区にわける
* [x] build yearを西暦に変える
* [x] 築何年を計算する
* [x] +Sを分離したものを付け加える
* [x] orderedの年と半期を分離する
* [x] layout_l1ごとにareaの平均.diffを取る -> あんまり改善せず
* [x] city codeを削除する
* [x] structureをmultihotに開く
* [x] area x floor_ratioで延べ床面積っぽいものができそう
* [x] 集約をmean以外も試す
* [x] catboostとlgbmをアンサンブルする
* [x] 部屋数
* [x] 一部屋当たりの面積
* [x] 何階立てか
* [x] diff ratioをもう一段集約する

## memo

* 本当は西暦が一年ずれているけど全部一年ずれているのでまあいいか
  * v5で直した
* price_logが小さいところで特に上振れた予測をして外している
* シングルモデルでは0.073台まで来ている(6/14現在)
* districtが同じで違う県、とかはある？->あったので、prefecture>city>districtを連結して置換した
* アンサンブルしても0.001上がるかなというかんじなのと、catboostが時間かかるので、シングルモデルの改善をしたほうがいいかも

## 疑問

* total_floor_areaとかは効きそうな散布図をしているが悪化するのはなぜ
* diff_groupでメモリが足りずにkilledになってしまう
* catb diff回すと固まる

## 参考

* https://static.signate.jp/competitions/182/summaries/3%E4%BD%8D.pdf
* https://naotaka1128.hatenadiary.jp/entry/kaggle-compe-tips


## tips

* ハイパーパラメータはdepthとmin_child_samplesが大事らしい。
  * depthはあげるとover fitting気味になり、min_child_samplesは上げるとunder_fitting気味になる