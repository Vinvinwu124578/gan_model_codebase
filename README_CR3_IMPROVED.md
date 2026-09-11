# CR3 采样改进版：路线预览与高度测量

本版增加预检复用、离线路线可视化及带验收条件的高度校正。**0.1 mm 是输入观测的误差预算上限，不是机器人绝对精度或“零误差”的保证。** 尚需现场确认实际托座、Tool/TCP、板面及视觉接触阈值。

## 安装与离线预览

Windows 建议把完整项目解压到短路径，例如 `C:\gan`，避免深层输出超过传统 260 字符路径限制。在该项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$py = ".\.venv\Scripts\python.exe"
$board = "outputs/tactile_gan_coverage_board_v4_highprotrusion_deepcontact_70mm_mountpitch150"
& $py tools/auto_cr3_v4_150mm_coverage_board_sampler.py --tile tile_nw --board-dir $board --dock-design hardware/raised_crossbar_preview_design.json --fixture-local-preview --profile quick --max-samples 12 --zero-tilt --output-dir preview_raised
```

打开 `preview_raised/fixture_local_motion_preview.html` 查看前 6 点的进出 TCP 路线，`planned_sampling_points_3d.html` 查看全部采样点；鼠标可旋转、缩放。JSON/CSV 保留全部计划参数。路线图中的网格和尖端轮廓用于示意，不是整机碰撞检测。

发布前已用实际提供的 STL 生成 2000 点计划，离线计划与 HTML 生成约 0.8 秒；这不是现场 IK 预检时间。生成的运行目录不提交 Git。复跑上述命令时，把 `--profile quick --max-samples 12 --zero-tilt` 换成 `--samples-per-tile 2000`，并使用新的输出目录。验证命令为 `python -m unittest discover -s tests -q`，本次 120 项离线测试全部通过。

`hardware/raised_crossbar_preview_design.json` **仅用于几何预览**，不能代替实际托座设计数据或机器人 fixture profile。上述命令不连接机器人、不执行 IK，也不验证机器人基座坐标中的可达性和碰撞。不要在该预览配置上添加执行、示教或高度标定参数。

## 已改进的预检与接触流程

- IK 缓存仅在一次不运动的预检内复用完全相同的查询；键包含完整 TCP、邻近关节分支、User 和 Tool，不舍入坐标。新增进度、缓存命中和耗时统计，收益取决于实际重复查询数量。
- STL 网格数据在生成预览时复用；`--skip-previews` 可省去网格和 HTML 生成，仍保留 JSON/CSV 计划及机器人检查。路线预览时不要使用此参数。
- 搜索访问明确终点；视觉接触在原位置进行多帧确认，并保存真实反馈首触区间，不以计划 TCP 冒充实测位置。

## 先纠正实际抬高横梁的几何基准

本次提供的抬高横梁 STL，其中央支承点在板局部坐标中为 **`(0, -139, 27) mm`**；这是按仓库毫米约定读取的模型几何，打印和安装后的尺寸仍需实测。

如果实际在这个 `z=27` 支承点记录 TCP，却在变换中仍使用旧托座的 `z=12` 名义坐标，则在相同坐标约定下，板面目标会沿板局部 +Z 偏高 **15 mm**。这足以解释“碰不到板”，但成立条件必须结合实际安装与 profile 核对，不能直接把全部任务统一减去 15 mm。

真机流程需要与该实物一致的完整 `dock_design.json`，包括横梁接触点、名义就座 TCP 和其他路线所需几何；还需要同一次安装、同一 User/Tool 下重新取得的真实 `fixture_profile.json`。旧设计或旧 profile 不可混用；STL 或预览 JSON 本身不能提供机器人基座坐标和触觉参考。

150 mm 入口脚本已取消旧 v1 托座的隐式默认值，改为抬高横梁配置路径。真实设计文件尚未补齐时，默认运行会因文件缺失而停止；离线预览按上面的命令显式指定预览配置。

## 高度测量与验收

在实际设计/profile 已核对、现场运动流程确认后，每次只测一个已知平面参考点。测量模式使用 `--height-measurement --first-contact-test --max-samples 1 --site <实际参考点ID> --skip-reference-pad-check`；此处仅说明必要参数，不提供使用预览设计的真机执行命令。

保持零倾角、零手动高度偏移，不加载已有高度校正，也不叠加 reference-pad 修正；关闭抖动。粗搜索先取得实际反馈定义的接触/未接触区间，区间须不超过 **1 mm**；然后仅在该已探索区间内以不超过 **0.02 mm** 的步进细测，不扩大下探边界。细测需重新确认释放、取得新图像并在同一位置确认接触；不满足条件则拒绝该测量。最终实际反馈区间须不超过 **0.04 mm**。测量模式不追加正常数据采样压入深度，并保存搜索图像。

标定点必须是 `stimulus=flat_reference`。边缘、曲面或纹理可能因 TacTip 的有限接触面积提前接触，不能据此拟合统一的板面高度残差。当前仓库只有 R01 的这类参考点，校正只覆盖所测 R01 点的凸包，且仅支持零倾角。**它不能校正或外推到整块板。** 全板校正需要现场分布更广、几何已知且可独立测量的平面参考区，并另行配置真实点位。

至少选择 **3 个不共线拟合点和 2 个不同位置的独立验证点**，验证点必须位于拟合点凸包内；每个点至少 **3 次独立测量**，每次保存一个新的 `collection.json`。可以用 R01 四角点拟合、另选内部点验证来改善覆盖；同点各次坐标必须完全相同。复制日志或把同一位置换个 ID 不算独立重复或验证。

## 导出 → 拟合 → 加载

下面全部为离线处理。路径应替换为真实设计、真实 profile 与本次测量文件；`fit` 和 `validate` 文件夹下按“点位/重复”分别保存采集子目录。

```powershell
$fixture = "C:\现场数据\实际托座_fixture_profile.json"
$design = "C:\现场数据\实际抬高横梁_dock_design.json"
$manifest = "$board/tactile_gan_coverage_board_340mm_manifest.json"
$fitFiles = Get-ChildItem ".\measurements\fit\*\collection.json"
$valFiles = Get-ChildItem ".\measurements\validate\*\collection.json"
$exportArgs = @("--fixture-profile", $fixture, "--dock-design", $design, "--board-manifest", $manifest, "--output", "measurements.csv")
foreach ($f in $fitFiles) { $exportArgs += @("--fit-collection", $f.FullName) }
foreach ($f in $valFiles) { $exportArgs += @("--validate-collection", $f.FullName) }
& $py tools/export_board_height_measurements.py @exportArgs
```

导出器核对三个文件的 SHA256、tile/User/Tool，拒绝缺失反馈区间、非平面点、校正叠加及坐标变化；横向反馈偏差默认上限为 **0.05 mm**，同样只是软件验收条件。旧日志没有所需实测字段时不能补造。

先用独立基准或量具测量并记录不确定度，覆盖视觉阈值相对物理首触的偏差及适用的 TCP/定位误差；记录仪器、方法、日期和证据。**不能把 0.02 mm 指令步长或反馈的小数位数填成测量不确定度。** 以下命令要求输入真实记录，User/Tool 必须与采集一致：

```powershell
$u = [double](Read-Host "请输入独立测量的不确定度上界，单位 mm")
$evidence = Read-Host "请输入仪器/基准、方法、日期及不确定度证据"
& $py tools/board_height_calibration.py --csv measurements.csv --output height_calibration.json --fixture-profile $fixture --dock-design $design --board-manifest $manifest --tile tile_nw --user 0 --tool 2 --measurement-uncertainty-mm $u --measurement-reference $evidence
```

默认验收：区间宽度 ≤0.04 mm、同点重复极差 ≤0.03 mm、拟合/独立验证残差 ≤0.03 mm；**最大残差 + 最大区间半宽 + 最大重复极差 + 独立测量不确定度 ≤0.10 mm**。不通过就检查测量或硬件，不应放宽参数来宣称满足 0.1 mm。程序也不覆盖既有校正文件。

通过后，使用同一真实配置，在计划命令中加入 `--height-calibration height_calibration.json --zero-tilt --skip-reference-pad-check`，并限定所有计划及备用点都在实测凸包内。程序会在任何相机/机器人操作前检查文件绑定、范围和预算；禁止手动偏移叠加与外推。托座重装、Tool/User、传感器或板发生变化后应重新测量。通过验收仍不等于真实接触处处达到 0.1 mm，更不等于没有误差。
