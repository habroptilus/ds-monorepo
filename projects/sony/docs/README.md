# Sonyコンペメモ

## EDAの気づき
* Cityがtrainとtestで重複がない。のでgroup kfoldが良さそう&Cityは予測には使わない
* Countryで集約特徴量を作ると精度改善
  * ratioやdiff_ratioをいれると悪化
* Cityで集約特徴量を作ると精度悪化(train+testでfitしているので実装は問題なさそう)
  * できた特徴量からCityが伝わってしまっている？

## 特徴量アイデア

* [] ラグ特徴量
* [x] monthで集約特徴量