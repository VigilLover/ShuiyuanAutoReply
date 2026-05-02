# Git 开发工作流

## 仓库结构

- **upstream**: `https://github.com/Hydroiodic/ShuiyuanAutoReply.git` （上游原始仓库）
- **origin**: `https://github.com/VigilLover/ShuiyuanAutoReply.git` （你的 fork）

| 分支 | 用途 |
|------|------|
| `main` | 纯净镜像，始终与 `upstream/main` 一致，不直接开发 |
| `dev` | 开发 + 日常运行，所有自定义功能在此 |

## 初始化（已完成）

当前仓库已配置完毕：
- `main` 跟踪 `upstream/main`
- `dev` 包含你的所有自定义提交，开发和运行都在此分支

## 日常开发

在仓库目录正常写代码、提交：

```bash
# 确保在 dev 分支
git checkout dev

# 写代码...
# 查看修改
git status

# 提交
git add .
git commit -m "feat: 你的改动描述"
git push origin dev
```

## 同步上游更新

当上游有新代码时：

```bash
# 1. 更新 main
git checkout main
git pull upstream main
git push origin main

# 2. 将 dev rebase 到最新的 main 上
git checkout dev
git rebase main

# 如果有冲突，手动解决后：
#   git add .
#   git rebase --continue
# 想放弃：
#   git rebase --abort

# 3. 推送
git push origin dev --force-with-lease
```

## 日常运行

直接在仓库目录启动程序：

```bash
cd ~/SoftwareTools/ShuiyuanAutoReply
git checkout dev
git pull origin dev
uv run python example/main.py
```

如果使用 cron 定时任务：

```bash
#!/bin/bash
cd ~/SoftwareTools/ShuiyuanAutoReply
git checkout dev
git pull origin dev
uv run python example/main.py
```

## 冲突最小化策略

1. **频繁同步**：至少每周 rebase 一次上游，差距越小冲突越少
2. **加新文件优先**：自定义功能尽量放独立文件中（如 `pet_model.py`），不修改上游核心文件
3. **标记自定义区域**：如果必须改公共文件，用注释标记：
   ```python
   # === CUSTOM: pet_model integration ===
   from pet_model import PetModel
   # === END CUSTOM ===
   ```
4. **配置优先**：能通过 `.env` / 环境变量控制的，不要改上游代码

## 速查

```bash
# 查看状态
git log main..upstream/main --oneline    # 上游新增了什么
git log main..dev --oneline              # dev 比上游多了什么

# 同步上游
git checkout main && git pull upstream main && git push origin main
git checkout dev && git rebase main && git push origin dev --force-with-lease

# 运行
uv run python example/main.py
```

## 注意事项

- `main` 只跟踪上游，永远不在上面开发
- `dev` 只有你一个人用，`--force-with-lease` 安全
- GitHub 上提示 "dev had recent pushes" 忽略即可，不要向 main 创建 PR
