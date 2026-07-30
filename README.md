# 小白拼写

小白拼写是一款本地运行的 Streamlit 单词跟打与拼写复习应用。它不是选择题工具，而是让你通过反复输入完整英文单词，记住字母顺序并建立键盘肌肉记忆。

## 功能

- 一天一个 CSV，可在历史词表之间随时切换
- 三种练习模式：看英跟打、首尾提示、中文盲拼
- 每个单词连续正确拼写 3 次，每次开始自动播放一次发音
- 当天错词优先，组内随机且本轮顺序稳定
- 新题自动聚焦输入框，无需先用鼠标点击
- 按一次 Enter 保存答案并直接进入下一词
- 下一题上方保留上一题反馈，错词继续显示逐字母核对
- 基于字符序列对齐的漏写、多写、错写提示
- 有道美音播放，并只在进入新题时尝试自动播放
- 每次有效答题实时写回当前 CSV
- 首次写回前自动备份，保存过程使用原子替换
- 本轮统计、错题回顾、只练错题和最新 CSV 下载
- 按需生成 AI 记忆卡：双语例句、用法、拼写提示与联想记忆
- AI 记忆卡同时显示中文释义和简明英文释义
- 批量 AI：一键覆盖当前 CSV 全部单词，已有卡片自动跳过，支持暂停与失败重试
- DictionaryAPI 提供可核验的英文释义与词源，和 AI 联想分开显示
- 复习浏览：搜索、筛选并逐词查看当前词表和 AI 记忆卡
- 已学单词本：跨日期去重累计练习记录，并关联长期 AI 学习档案
- NVIDIA 主模型失败时自动尝试备用模型，结果写入本地缓存
- 词表支持二次确认删除：云端永久删除，本地移入可恢复目录

## 环境

- Python 3.10+
- macOS 为主要使用环境，也支持 Windows 和 Linux

## 安装

在 PyCharm 终端或系统终端中进入项目目录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 启动

```bash
streamlit run app.py
```

终端会显示本地地址，通常为 `http://localhost:8501`。浏览器没有自动打开时，手动访问该地址即可。

### macOS Dock 一键启动

项目包含 `macos/小白拼写.app` 和对应的 macOS `launchd` 服务配置。
安装后，Dock 中的“小白拼写”会通过系统后台服务启动 Streamlit 并打开页面；
重复点击不会重复启动服务。启动日志保存在
`.runtime/streamlit-launchd.log`，其中不应写入 API 密钥。

当前用户的安装位置为：

- 应用：`~/Applications/XiaobaiSpelling.app`
- 后台服务：`~/Library/LaunchAgents/com.zihaobai.xiaobaipinxie.plist`

## 每日使用

1. 从背词软件导出当天的 CSV。
2. 把文件放入 `vocabulary_data/`，也可以在应用左侧手动上传。
3. 启动应用；小白拼写默认选择最新修改的 `export` 文件。
4. 选择练习模式并开始输入。
5. 每次有效提交后，累计数据会自动写回当前 CSV。
6. 第二天把新的 CSV 放入 `vocabulary_data/`，点击“刷新文件列表”即可。

## AI 记忆助手

在 `.streamlit/secrets.toml` 中配置：

```toml
NVIDIA_API_KEY = "你的本地密钥"
NVIDIA_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
NVIDIA_FALLBACK_MODEL = "deepseek-ai/deepseek-v4-flash"
```

记忆助手不会在每次按 Enter 时自动调用。展开当前单词下方的
“AI 记忆助手”，点击“生成本词记忆卡”才会查询 DictionaryAPI 并调用
NVIDIA；成功结果保存在 `learning_cache/`，相同单词再次打开时直接读取缓存。

需要一次生成当天全部内容时，在左侧进入“批量 AI”，点击
“为当前词表生成全部 AI 记忆卡”。程序会先检查 Supabase 与本地缓存，
只为缺失单词逐个生成，完成一张保存一张。任务可以暂停；连续失败三次会
自动暂停。云端同步失败的卡片可单独重试，重试会复用本地结果，不会重复
调用模型。关闭页面后再次进入并重新扫描，也会跳过已经保存成功的卡片。

如果主模型端点失效或下线，程序会自动尝试备用模型。DictionaryAPI 未返回
词源时，界面会明确说明“AI 联想不等于真实词源”，不会让模型自行编造词根。

程序只扫描：

- 项目根目录中的 `*.csv`
- `vocabulary_data/*.csv`
- `uploaded_csv/*.csv`

不同日期的 CSV 不会合并，未选择的文件不会被读取或修改。

## Supabase 多设备云同步

云同步是可选功能。没有配置 Supabase 时，应用继续使用原来的本地 CSV；
配置成功后，上传、答题进度和 AI 记忆卡会同步到云端。

1. 在 Supabase 项目的 **SQL Editor** 中运行
   [`docs/supabase_schema.sql`](docs/supabase_schema.sql)。
2. 在 **Settings → API Keys** 中创建或复制 `sb_secret_` 开头的
   Secret key。Secret key 只能保存在服务器 Secrets 中，不能放进 GitHub、
   浏览器代码或截图。
3. 在本地 `.streamlit/secrets.toml` 中增加：

```toml
SUPABASE_URL = "https://你的项目编号.supabase.co"
SUPABASE_SECRET_KEY = "你的 sb_secret_ 服务器密钥"
```

4. 重启应用。左侧显示“Supabase 已连接”后，可点击
   “将当前本地词表导入云端”，也可以直接上传新 CSV。

升级旧项目时也可以安全地重新运行同一份 SQL。升级完成后进入
“已学词本”，点击“从历史云端词表重建”一次，即可把已有云端 CSV
回填为长期词汇档案。同一份词表按 ID 覆盖同步，重复执行不会重复累计。

以后上传的新 CSV 会自动更新已学单词本；只有 `当天答题次数 > 0` 的单词
会进入长期词本。同一个英文单词跨日期合并，但每天的记录分别保存，因此
可以准确计算首次学习、最近复习、学习天数、累计正确与错误次数。

云端词表使用版本号进行乐观锁定：如果两台设备同时修改同一份词表，
较晚提交的设备会收到重新加载提示，不会静默覆盖已经同步的进度。

## CSV 格式

CSV 必须包含以下字段：

```text
单词
中文释义
类型
当前状态
当天答题次数
当天正确
当天错误
```

支持 `utf-8-sig`、`utf-8` 和 `gb18030` 编码。推荐导出为 UTF-8 CSV。

## 数据安全

- 每次有效答题都会保存，而空答案不会计分或写盘。
- 当前文件第一次写入前，会在 `.backups/` 创建一次带时间戳的备份。
- 保存先写入同目录临时文件，再使用原子替换更新当前 CSV。
- 保存失败时，本题计数会回滚并显示原因，不会假装保存成功。
- 完成页可以下载内存中的最新 CSV，作为额外备份入口。
- `当前状态` 只读取，不由小白拼写修改。

请避免在答题时用其他程序同时覆盖当前 CSV。如果文件被外部删除，可先下载应用内存中的副本，再刷新文件列表。

## 测试

```bash
python -m compileall app.py
pytest -q
```

更完整的产品与实现约束见 [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md)。
