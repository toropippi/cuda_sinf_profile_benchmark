# プロファイリングベンチマーク

このリポジトリは、CUDA を用いて `__sinf()` が `MUFU.SIN` 命令に展開される仕組みを学習し、SFU（XU）と FMA のオーバーラップを確認するためのサンプルコードを提供します。

---

## ファイル構成

* `sincos_bench.cu`：ベンチマーク用 CUDA ソースコード
* `README.md`：本ドキュメント

---

## 動作環境

* CUDA Toolkit 11 以降
* NVIDIA GPU（例：RTX シリーズ）
* Linux / Windows

---

## ビルド方法

```bash
# GPU アーキテクチャ（sm_75 など）を適宜変更してビルド
nvcc -O3 -arch=sm_75 sincos_bench.cu -o sincosBenchCuda
```

---

## 実行方法

```bash
./sincosBenchCuda <mode> <iterations> [device]
```

| mode | 内容                              |
| :--: | :------------------------------ |
|   1  | Sin ＋ Mandelbrot（交互に 8 回）ベンチマーク |
|   2  | Sin 関数のみ（8 回ループ）                |
|   3  | Mandelbrot のみ（8 回ループ）           |
|   4  | Cos 関数のみ（1 回）                   |

例:

```bash
./sincosBenchCuda 1 1000000      # モード1、反復回数を 1,000,000 回
./sincosBenchCuda 4 100000 1    # モード4、デバイス ID 1 指定
```

---

## プロファイリング手順

1. NVIDIA Nsight Compute (ncu) あるいは Nsight Systems を起動
2. ベンチマークバイナリをプロファイリング実行して報告書（`.ncu-rep` など）を生成
3. レポート内の `MUFU.SIN` カウントと FMA カウントを確認
4. SFU（XU）と FMA ユニットが重なる実行サイクルを可視化してパイプライン隠蔽を検証

---

## 学習ポイント

* `__sinf()` がハードウェアで `MUFU.SIN` 命令に置き換わる仕組み
* SFU（XU）と FMA の命令オーバーラップによるレイテンシ隠蔽
* NVIDIA GPU のパイプライン構造とスコアボード挙動
* Nsight Compute の基本的な使い方と解析ポイント

---

## ライセンス

個人学習目的のサンプルコードです。自由に利用・改変してください。
