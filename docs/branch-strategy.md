# Fork 仓库的 main 主线与上游同步

自己的 fork 为 VigilLover/ShuiyuanAutoReply，上游为 Hydroiodic/ShuiyuanAutoReply。**当前工作流已统一从 main 发布；dev 保留为开发集成分支。** 本次只修改仓库配置，没有执行远程分支修改、默认分支设置、推送或合并。

## 分支职责

| 引用 | 角色 |
|---|---|
| origin/main | 自己的产品主线、GitHub 默认分支、正式发布来源 |
| origin/dev | 开发集成分支，通过 PR 合入 main |
| upstream/main | 原作者仓库主线，通过 git fetch upstream 跟踪 |

origin/main 和 upstream/main 属于不同 remote，同名不冲突。自己的 main 无需一直保留为原作者代码的镜像。

当前 Release 验证标签对应的提交属于 origin/main 的历史，发布说明标注 main；手动 Deploy 仅从 main 执行。CI 同时检查 dev 和 main。新提交若只存在于 dev，尚未进入 main，就不能创建正式发布。

## 首次切换操作

1. 把完整部署改造和本次配置提交到 dev。
2. 创建 dev → main 的 PR，等待 CI 通过后合并。建议 Create a merge commit，保留现有开发提交历史。
3. 在 GitHub 确认 main 为默认分支，为 main 设置必需检查 `Python (remote)`、`Python (local)`、`frontend` 及 PR 保护规则。
4. 合并完成并且 main 的 CI 通过后，再从 origin/main 打新版本标签。

操作前先检查工作区，不要使用 reset --hard 或 git add . 覆盖或混入其他工作：

```bash
git status --short
git fetch origin
git fetch upstream
git rev-list --left-right --count origin/main...origin/dev
```

若左侧为 0，main 没有独有提交，可以正常合并；若不是 0，应审查并通过合并保留双方历史，不强推覆盖。需要留档时可为迁移前的 main 创建一个未使用过的 upstream-baseline 分支名；它只是快照，不会自动同步上游。

无需重命名或删除 main。也不要只把默认分支改为 dev：当前 Deploy 明确限制 main，默认分支和发布来源应保持一致。

## 正式发布与后续开发

目标提交合入 main 后：

```bash
git fetch origin
git tag v1.0.0 origin/main
git push origin v1.0.0
```

标签已存在时使用新的版本号，不能移动已有正式标签。Release 成功后，在 Actions → Deploy production 中选择 main、输入版本号，显式触发部署。

可以继续在 dev 开发，再通过 dev → main PR 发布；也可以从 main 创建短期功能分支，经 PR 合入 main。不再需要“只向 main 同步工作流”的特殊流程。

分支配置变更不会修改服务器目录、镜像仓库、Cookie、数据库或向量空间。首次服务器配置见 [逐步部署指南](first-deployment.md)，发布失败和恢复策略见 [CI/CD 运维指南](cicd.md)。

## 同步原作者的更新

在干净工作区中，从自己的 main 创建独立同步分支：

```bash
git fetch origin
git fetch upstream
git switch -c sync/upstream-20260908 origin/main
git merge upstream/main
```

分支名改为实际同步日期，若已存在则换新名称。解决冲突、通过 CI，再把同步分支通过 PR 合入自己的 main。不要把自己的 main 重置为 upstream/main。

参考：[GitHub 设置默认分支](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch)。
