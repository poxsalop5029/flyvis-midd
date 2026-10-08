# `flyvis_midd` API

この文書は `src/flyvis_midd` に定義されている関数と定数をまとめたものです。

## インストールと依存関係

プロジェクトのルートで開発用インストールを行います。

```bash
pip install -e .
```

パッケージ名は `flyvis-midd`、Python からの import 名は `flyvis_midd` です。基本依存は NumPy です。描画モジュールには `matplotlib`、`flow`、`rendering`、`evaluation` モジュールには PyTorch が必要です。図表用の追加依存は次のように導入できます。

```bash
pip install -e '.[figures]'
```

## パッケージ直下から使える API

以下は `import flyvis_midd` で利用できます。

### MIDD 刺激

- `build_3midd_sequences(c, n_units=24) -> dict[str, numpy.ndarray]` — 3MIDD の `original`、`mirror`、`negative` の輝度列を `uint8` 配列で返します。
- `build_4midd_sequences(c1, c2, n_units=24) -> dict[str, numpy.ndarray]` — 4MIDD の3種類の輝度列を返します。
- `directed_pair_counts(sequence) -> collections.Counter` — 列を循環列として扱い、隣り合う有向輝度ペアを数えます（末尾から先頭へのペアを含む）。
- `render_sector_image(sequence, *, image_size=382, inner_diameter=55, outer_diameter=382, phase_deg=0.0) -> (image, ring_mask)` — 輝度列からリング状のセクター画像とリングのブールマスクを生成します。画像は `uint8` の `(H, W)` 配列です。
- `load_midd_sequences() -> numpy.ndarray` — Fig. 1B 用の6条件を返します。形状は `(6, 50, 382, 382)`、値域は `[0, 1]` の `float32` です。

### プロペラ刺激

- `PROPELLER_CONDITIONS` — 時計回り／反時計回り各3速度と静止条件を定義した順序付き辞書です。
- `make_propeller_frames(n_frames=50, fps=50, size=128, n_blades=3, speed_hz=1.0, clockwise=True, static=False, hub_radius=5, blade_width=0.20) -> numpy.ndarray` — プロペラ動画を `(frame, y, x)` の `float32` 配列で生成します。
- `make_propeller_stimulus_set(conditions=None, **frame_kwargs) -> numpy.ndarray` — 条件ごとの動画を積み重ねた配列を返します。条件辞書を省略すると `PROPELLER_CONDITIONS` を使います。
- `estimate_propeller_phase(frame, n_blades=3) -> float` — 画像モーメントからブレード位相をラジアンで推定します。

## モジュール別 API

### `flyvis_midd.flow`

フロー推定、座標変換、空間サンプルの集約を行います。このモジュールの関数はパッケージ直下には再 export されていないため、`from flyvis_midd.flow import ...` の形式で読み込みます。PyTorch が必要です。

- `canonical_flow_layout(flow)` — 4次元フローを `(samples, frames, 2, hexels)` に揃えます。成分軸を認識できない場合は `ValueError`。
- `run_flow_inference(luminance, network, decoder, dt=1/50, fade=1.0, device=torch.device("cpu"))` — 入力をfloat32へ変換し、network/decoderを `.to(device).eval()` で再帰的にevaluation modeへ置きます。`torch.no_grad()` 内で `fade_in_state(fade, dt, luminance[:,0])` → `network.simulate(..., initial_state=...)` → `decoder(responses)` を実行します。fade-inフレームは記録動画へ連結せず、`canonical_flow_layout()` で揃えた未変換のdecoder `(u,v)` とluminance、responsesをCPU tensorの辞書で返します。FlyVisは別途インストールします。
- `extract_cell_type(responses, network, cell_type)` — `LayerActivity` から指定した細胞型の応答を取り出します。
- `flow_uv_to_polar(flow, flip_y_for_display=True, eps=1e-8)` — `(radius, theta_deg, valid_theta)` を返します。表示座標系では既定で y 成分の符号を反転します。
- `map_normal_positions_to_layout(receptors, layout)` — 通常の BoxEye 受容体中心をレイアウト画像座標に写し、座標とスケールを返します。
- `merge_duplicate_vectors(yx, vectors)` — `(frames, positions, 2)` のベクトル場で重複座標を平均します。
- `merge_spatial_samples(yx, *samples)` — 座標の重複を平均し、`(unique_yx, *merged_samples, counts)` を返します。各サンプルの先頭軸が座標数と一致する必要があります。
- `merge_spatial_axis(yx, values, position_axis=-2)` — 任意の値配列について指定軸上の重複座標を平均します。
- `merge_layout_field(centers_yx, luminance, flow)` — レイアウト上の重複座標を統合し、固有座標、輝度 `(frames, N)`、フロー `(frames, N, 2)` を返します。
- `native_boxeye_field(receptors, luminance, flow)` — BoxEye のネイティブ座標、輝度、正準化したフローを返します。
- `flow_roi_metrics(centers_yx, flow, *, baseline_flow, frame_slice=slice(None), direction_deg, radius_fraction=0.65, flip_y_for_display=True)` — ベースライン差分フローの中央円形 ROI 指標を辞書で返します。
- `idw_resample(source_yx, source_vectors, target_yx, k=6, eps=1e-8)` — 逆距離重み付けでベクトルを補間し、補間値・距離・重み・近傍インデックスを返します。

### `flyvis_midd.evaluation`

Fig.1/2のsingle-BoxEye MIDD評価を共有するAPIです。NumPyとPyTorchを使用します。FlyVis本体は推論時に別途インストールし、保存flowの解析や軽量testsには不要です。

| API | 入力・目的 | 出力・既定値 |
|---|---|---|
| `MIDDEvaluationConfig` | timing、BoxEye設定、解析窓、annulus、座標変換を明示する不変dataclass | 下記canonical defaults。`.provenance()` は設定と符号規約をJSON互換の辞書で返す。設定だけではrendererを実行しない。 |
| `CANONICAL_MIDD` | 共通評価用の `MIDDEvaluationConfig()` インスタンス | 下記default contract。 |
| `analyze_midd_flow(flow, centers_yx, *, config=CANONICAL_MIDD, flow_coordinates="decoder")` | raw decoder flowを `(1, frames, 2, receptors)` へ揃え、指定窓とannulusで測定。`centers_yx` は有限な `(receptors,2)` の画像座標。単一spatial sampleのみ対応。 | `mean_signed_tangent`（T）、magnitude、ratio、radial成分、annulus件数とmaskの辞書。既定は50フレーム中10:50。NumPy配列またはCPUへ移動可能なtensorを受け取る。 |
| `aggregate_midd_metrics(metrics_by_variant)` | `original`、`mirror` と任意の `negative`。各値は有限scalar／同じshapeの配列、または `mean_signed_tangent` を含むmetric辞書。 | `T_original`, `T_mirror`, `A`, `abs_A`, `S`, `abs_S`, `M`, `actual_reversal`。negativeがあれば独立に `T_negative` を返す。scalar入力にはPython scalarを返す。 |

#### Canonical MIDD contract

```text
dt = 1/50 = 0.02 s
fade-in = 1.0 s, stimulus frame 0; fade frames are not concatenated
50 recorded stimulus frames; analysis = Python slice 10:50 (frames 10–49)
BoxEye extent = 15; kernel_size = 13
annulus = 0.14–0.98 × r_max
raw decoder coordinates = (u,v)
image conversion = (u,-v), exactly once inside the rotation metric
image axes: x rightward, y downward; rotation = CW positive

A = (T_original - T_mirror)/2
S = (T_original + T_mirror)/2
M = abs(A) - abs(S)
actual_reversal = T_original*T_mirror < 0 (equivalently M > 0)
```

Negative controlはA/S/M/reversalへ使用しません。`analyze_midd_flow` へ渡す前にvを反転しないでください。`flow_coordinates="image"` や `flip_decoder_y=False` は拒否します。ただし配列値だけでは、申告されない事前反転を検出できません。保存済み変換後scalarはこのflow APIへ入力せず、`aggregate_midd_metrics` へ直接渡します。

```python
from flyvis_midd.evaluation import CANONICAL_MIDD, analyze_midd_flow, aggregate_midd_metrics
from flyvis_midd.flow import run_flow_inference

# network, decoder, receptor centers and rendered condition movies are supplied by the caller.
metrics = {}
for variant, luminance in movies_by_variant.items():
    result = run_flow_inference(luminance, network, decoder,
                                dt=CANONICAL_MIDD.dt, fade=CANONICAL_MIDD.fade_seconds)
    metrics[variant] = analyze_midd_flow(result["flow"], centers_yx)
paired = aggregate_midd_metrics(metrics)
```

### `flyvis_midd.rendering`

BoxEye の画像レンダリングとハニカムレイアウトへの画像配置を行います。画像や数値は主に PyTorch テンソルとして扱います。

- `as_gray_tensor(image)` — 入力画像を CPU 上の `float32` グレースケール `(H, W)` テンソルに変換します。`uint8` は `[0, 1]` に正規化します。
- `resize_to_boxeye_frame(image, receptors)` — 画像を BoxEye の必要フレームサイズにリサイズします。
- `manual_crop_means(image, receptors)` — 受容体カーネルごとの平均輝度を手動計算します。
- `compare_renderers(image, receptors, *, resize=False, atol=1e-6)` — BoxEye の出力と手動計算を比較し、値、差分、一致判定、誤差統計を辞書で返します。
- `make_layout_geometry(row_counts, receptors)` — 行ごとのフィールド数と受容体配置からレイアウト座標・境界などの幾何情報を構築します。
- `place_image_in_layout(frame, geometry, *, fit_mode="preserve_aspect")` — 画像をレイアウトに配置し、配置画像と原点を返します。`fit_mode` は `preserve_aspect`、`inscribed_square`、`full_vertical`。
- `crop_means_at_centers(image, centers_yx, origin_yx, kernel_size, *, padding_mode="constant", padding_value=0.0)` — 任意の中心座標で局所平均を計算します。
- `render_layout_frame(frame, receptors, row_counts, geometry=None, *, fit_mode="preserve_aspect", padding_mode="constant", padding_value=0.0)` — 1フレームをレイアウトに描画し、受容体値・画像・座標情報を辞書で返します。
- `render_layout_video(video, receptors, row_counts=(3, 2, 3), *, fit_mode="preserve_aspect", padding_mode="constant", padding_value=0.0)` — `(frame, height, width)` 動画を描画し、値列とレイアウト情報を返します。
- `image_bounds(image_origin_yx, image_shape_yx)` — 配置画像の描画用境界を辞書で返します。

### `flyvis_midd.plotting`

Matplotlib による可視化関数です。`figures` 追加依存（Matplotlib）が必要です。

- `show_renderer_comparison(result, title)` — 入力画像、BoxEye 値、手動値、差分を並べて表示し、`(fig, axes)` を返します。
- `plot_layout_rendering(ax, result, values, title, draw_outer_frame=True)` — レイアウトの受容体値と境界を指定軸に描画します。
- `draw_hex_flow(ax, yx, values, flow_uv, title, color="red", stride=8, size=12, scale=None, width=0.0035, axis_off=False)` — 六角形サンプルとフローベクトルを指定軸に描画します。

## import 例

```python
from flyvis_midd import build_3midd_sequences, make_propeller_frames
from flyvis_midd.flow import flow_uv_to_polar
from flyvis_midd.rendering import render_layout_video
from flyvis_midd.plotting import draw_hex_flow

sequences = build_3midd_sequences(48)
movie = make_propeller_frames(n_frames=50, speed_hz=1.0)
```

### `flyvis_midd.rotation`

CW-positive な回転フロー指標を計算します。元実装は `fvsp/src/flyvis_sparsity/rotation.py` から移しています。NumPy が必要です。

- `rotation_basis(centers_yx, center_yx=None)` — 中心座標を基準に半径、動径単位ベクトル、接線単位ベクトルを返します。
- `rotation_flow_metrics(centers_yx, flow_uv, *, center_yx=None, inner_radius_fraction=0.14, outer_radius_fraction=0.98, flip_decoder_y=True, flip_y_for_display=None)` — CW 正方向で動径・接線成分などの回転フロー指標を計算します。
- `image_rotation_flow_metrics(...)` — 画像座標（y が下向き）に合わせた同等の指標を返します。`flow_uv` の末尾2軸は `(positions, 2)` です。ノートブックのモデル選択セルではこの関数の `mean_signed_tangent` を使います。
