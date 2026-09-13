# Radeon Pro W7900 在 Linux 下的风扇控制：排查记录与 systemd 配置

本文记录 Radeon Pro W7900（Navi 31，PCI ID `1002:7448`）在 Arch Linux 上升级内核和 AMDGPU 固件后，风扇控制接口消失或设置被恢复的问题，以及当前已经验证可用的解决方案。

当前方案实现以下行为：

- 使用 GPU junction（hotspot）温度；
- 温度达到或超过 70°C 时，将最低风扇 PWM 提交为 100%，使风扇拉满；
- 温度下降到或低于 65°C 时，将最低风扇 PWM 恢复为 20%；
- 65–70°C 之间保持当前状态，利用 5°C 回差避免反复切换；
- 每 2 秒检查一次，并在 GPU 重置导致设置丢失时重新应用；
- 服务停止时向驱动发送 `r`，恢复固件默认设置。

## 硬件与已验证的软件组合

排查时使用的硬件和路径如下：

```text
GPU: Radeon Pro W7900 / Navi 31
PCI 地址: 0000:03:00.0
PCI ID: 1002:7448
子系统 ID: 1002:0e0d
驱动: amdgpu
```

最终恢复控制能力的组合：

```text
linux-zen: 6.19.6.zen1-1
运行内核: 6.19.6-zen1-1-zen
linux-firmware-amdgpu: 20260622-1
LACT: 0.10.1-1
```

内核命令行包含：

```text
amdgpu.ppfeaturemask=0xffffffff
```

可用组合下会出现以下目录：

```text
/sys/bus/pci/devices/0000:03:00.0/gpu_od/fan_ctrl/
```

其中最重要的接口是：

```text
fan_minimum_pwm
```

本机报告的有效范围为 20–100：

```text
FAN_MINIMUM_PWM:
20
OD_RANGE:
MINIMUM_PWM: 20 100
```

## 现象与排查结论

最初手动执行以下命令可以让风扇拉满，但过一会设置会恢复为 20：

```bash
echo 100 > fan_minimum_pwm
echo c > fan_minimum_pwm
```

这里的 `100` 是待提交的新值，`c` 表示 commit。`fan_minimum_pwm` 修改的是固件自动控扇策略中的最低 PWM，并不是直接切换到固定转速模式。将最低值设置成 100%，实际效果仍然是自动模式，但固件可以使用的最低速度已经是满速。

当时系统中运行着 `lactd.service`。LACT 会在检测到 DRM 设备事件后重新加载 GPU 并应用自己的配置，因此直接在 sysfs 中临时写入的值可能被后续配置覆盖。需要确保同一时间只有一个组件负责风扇策略。

升级到以下组合后，问题发生了变化：

```text
内核: 7.2.4-zen2-1-zen
linux-firmware-amdgpu: 20260910-1
```

此时整个 `gpu_od/fan_ctrl` 目录消失。通过 `hwmon` 尝试手动控制也失败：

```bash
echo 1 > pwm1_enable
echo 255 > pwm1
```

实际表现是 `pwm1_enable` 仍读回 `2`（自动模式），写 `pwm1` 返回 `Invalid argument`。停止 LACT、保持 GPU 设备打开以及关闭 PCI 运行时休眠后，结果仍然相同，因此这次失败不是 LACT 覆盖或 GPU 休眠单独造成的。

随后测试 `6.18.51-1-lts`，但继续使用 `linux-firmware-amdgpu 20260910-1` 时，`gpu_od/fan_ctrl` 仍未恢复。最后同时使用 `6.19.6-zen1-1-zen` 和缓存中的 `linux-firmware-amdgpu 20260622-1`，接口恢复，`fan_minimum_pwm` 再次可以正常提交。

根据这组对比，AMDGPU 固件版本是关键变量，内核版本也可能参与接口能力判断。它证明了当前可用组合，但不能仅凭这些实验断言所有 W7900 都由同一个固件改动导致。

## 为什么没有直接使用 LACT 风扇曲线

在当前 W7900 上，`fan_curve` 文件存在，但驱动报告的允许范围错误地是 0°C–0°C、0%–0%：

```text
OD_FAN_CURVE:
0: 0C 0%
1: 0C 0%
2: 0C 0%
3: 0C 0%
4: 0C 0%
OD_RANGE:
FAN_CURVE(hotspot temp): 0C 0C
FAN_CURVE(fan speed): 0% 0%
```

因此 LACT 无法通过 PMFW 曲线接口写入普通曲线，日志会出现：

```text
Invalid fan curve: Temperature 40℃ is outside of the allowed range 0℃ to 0℃
```

LACT 的固定转速模式和直接写 `hwmon/pwm1` 在不可用的内核/固件组合上也会返回 `EINVAL`。最终采用独立 systemd 服务，通过已验证可用的 `fan_minimum_pwm` 实现 70°C 阶梯策略。

为避免冲突，LACT 保留其他 GPU 设置，但必须关闭其风扇控制：

```yaml
version: 7
daemon:
  log_level: info
  admin_group: wheel
  disable_clocks_cleanup: false
apply_settings_timer: 5
gpus:
  1002:7448-1002:0E0D-0000:03:00.0:
    fan_control_enabled: false
    performance_level: manual
    power_profile_mode_index: 0
current_profile: null
auto_switch_profiles: false
```

## 完整温控脚本

脚本安装路径：

```text
/usr/local/sbin/w7900-fan-step
```

完整内容：

```bash
#!/usr/bin/env bash
set -u

device=/sys/bus/pci/devices/0000:03:00.0
control="$device/gpu_od/fan_ctrl/fan_minimum_pwm"
high_temperature=70000
low_temperature=65000
normal_pwm=20
full_pwm=100
state=unknown

log() {
    logger -t w7900-fan "$*"
}

apply_pwm() {
    local pwm=$1
    printf '%s\n' "$pwm" > "$control" || return 1
    printf 'c\n' > "$control" || return 1
    log "minimum PWM set to $pwm%"
}

restore_default() {
    if [[ -w "$control" ]]; then
        printf 'r\n' > "$control" 2>/dev/null || true
    fi
}

trap 'restore_default; exit 0' TERM INT

while true; do
    temperature=
    for candidate in "$device"/hwmon/hwmon*/temp2_input; do
        if [[ -r "$candidate" ]]; then
            temperature=$candidate
            break
        fi
    done

    if [[ -n $temperature && -w "$control" ]] && read -r temp < "$temperature"; then
        desired=
        if (( temp >= high_temperature )); then
            state=full
            desired=$full_pwm
        elif (( temp <= low_temperature )); then
            state=normal
            desired=$normal_pwm
        elif [[ $state == full ]]; then
            desired=$full_pwm
        elif [[ $state == normal ]]; then
            desired=$normal_pwm
        fi

        if [[ -n $desired ]]; then
            current=$(sed -n '2p' "$control" 2>/dev/null || true)
            if [[ $current != "$desired" ]]; then
                apply_pwm "$desired" || log "failed to set minimum PWM to $desired%"
            fi
        fi
    fi
    sleep 2
done
```

脚本不依赖不稳定的 `hwmon2` 编号，而是在固定的 PCI 设备目录下动态寻找 `hwmon*/temp2_input`。本机的 `temp2_label` 是 `junction`。在其他机器上部署前应先确认：

```bash
for h in /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*; do
    cat "$h/temp2_label" "$h/temp2_input"
done
```

温度接口使用毫摄氏度，因此 70°C 写作 `70000`，65°C 写作 `65000`。

## 完整 systemd 单元

单元安装路径：

```text
/etc/systemd/system/w7900-fan-step.service
```

完整内容：

```ini
[Unit]
Description=Radeon Pro W7900 70C full-speed fan control
After=multi-user.target lactd.service

[Service]
Type=simple
ExecStart=/usr/local/sbin/w7900-fan-step
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

`After=lactd.service` 只规定启动顺序。LACT 中的 `fan_control_enabled` 仍需设为 `false`，否则两个组件可能互相覆盖设置。

## 安装与启用

将脚本和 unit 文件放到上述路径后执行：

```bash
sudo chmod 755 /usr/local/sbin/w7900-fan-step
sudo chmod 644 /etc/systemd/system/w7900-fan-step.service
sudo systemctl daemon-reload
sudo systemctl enable --now w7900-fan-step.service
```

查看状态：

```bash
systemctl status w7900-fan-step.service
systemctl is-enabled w7900-fan-step.service
```

查看温控服务的切换记录：

```bash
journalctl -t w7900-fan -f
```

查看当前温度、最低 PWM、实际 PWM 和 RPM：

```bash
device=/sys/bus/pci/devices/0000:03:00.0
hwmon=$(find "$device/hwmon" -mindepth 1 -maxdepth 1 -type d | head -n1)

cat "$hwmon/temp2_input"
cat "$device/gpu_od/fan_ctrl/fan_minimum_pwm"
cat "$hwmon/pwm1"
cat "$hwmon/fan1_input"
```

经过实际验证，提交最低 PWM 100% 后：

```text
pwm1: 255
fan1_input: 约 3745 RPM
```

将最低 PWM 恢复到 20% 后，风扇不会瞬间下降，而是由固件逐渐减速。这是正常现象。

## 固件回退与版本锁定

本机 pacman 缓存中保留了已验证版本，因此使用以下命令回退：

```bash
sudo pacman -U /var/cache/pacman/pkg/linux-firmware-amdgpu-20260622-1-any.pkg.tar.zst
```

回退后需要重启，因为正在运行的 GPU 已经加载了旧版本或新版本固件，替换磁盘文件不会让固件立即重新加载。

为了防止下一次全系统升级重新安装不可用版本，在 `/etc/pacman.conf` 的 `[options]` 中加入：

```ini
# Keep the AMD GPU firmware version that retains W7900 fan control.
IgnorePkg = linux-firmware-amdgpu
```

这会让 pacman 在升级时跳过该包。以后确认新固件已经修复时，应删除 `IgnorePkg` 中的这个包名，再正常升级并重新验证接口。

## 调整阈值

修改脚本中的以下变量：

```bash
high_temperature=70000
low_temperature=65000
normal_pwm=20
full_pwm=100
```

修改后重启服务：

```bash
sudo systemctl restart w7900-fan-step.service
```

`normal_pwm` 和 `full_pwm` 必须位于 `fan_minimum_pwm` 输出的 `MINIMUM_PWM` 范围内。不同显卡或不同固件的范围可能不同。

## 停用与回滚

停止并禁用服务：

```bash
sudo systemctl disable --now w7900-fan-step.service
```

服务收到停止信号后会向 `fan_minimum_pwm` 写入 `r`，请求驱动恢复固件默认值。随后可以删除文件：

```bash
sudo rm /etc/systemd/system/w7900-fan-step.service
sudo rm /usr/local/sbin/w7900-fan-step
sudo systemctl daemon-reload
```

若要恢复 AMDGPU 固件自动升级，从 `/etc/pacman.conf` 的 `IgnorePkg` 中删除 `linux-firmware-amdgpu`。

## 排查命令速查

确认内核、固件和驱动：

```bash
uname -r
pacman -Q linux-zen linux-firmware-amdgpu
lspci -nnk -s 03:00.0
cat /proc/cmdline
```

确认风扇接口：

```bash
device=/sys/bus/pci/devices/0000:03:00.0
ls -l "$device/gpu_od/fan_ctrl"
cat "$device/gpu_od/fan_ctrl/fan_minimum_pwm"
```

检查服务冲突和错误：

```bash
systemctl status lactd w7900-fan-step
journalctl -u lactd -u w7900-fan-step -b --no-pager
ps -eo pid,comm,args | rg -i 'lact|corectrl|fancontrol|amdgpu-fan|radeon-profile'
```

检查 GPU reset、SMU 和固件信息：

```bash
journalctl -k -b --no-pager | rg -i 'amdgpu|smu|reset|fan'
```

如果 `fan_minimum_pwm` 可以读写但数值自动恢复，优先检查后台控制服务。如果整个 `gpu_od/fan_ctrl` 不存在，应优先核对内核与 `linux-firmware-amdgpu` 版本组合，而不是继续尝试向不存在的接口写值。
