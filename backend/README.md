# 网络欺凌识别与干预 FastAPI 服务

## 三项功能

1. 使用训练好的 DistilBERT 判断是否属于网络欺凌，并返回主类型、多标签类型和概率。
2. 通过 LLM 提取原文关键片段；服务端校验片段、计算字符位置并生成安全高亮 HTML。
3. 通过 LLM 生成温和、有效、适合青少年的中文干预提醒。
4. SQLite 用户注册、登录和 JWT 鉴权；文本分析接口仅对登录用户开放。

## 安装与启动

```powershell
python -m pip install -r backend/requirements.txt
$env:LLM_API_KEY="你的密钥"
$env:LLM_MODEL="gpt-4.1-mini"
$env:JWT_SECRET="请设置至少32位的随机字符串"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

也可以将配置写入 `backend/.env`。服务启动时会自动加载该文件；如果同名
系统环境变量已经存在，则以系统环境变量为准。修改 `.env` 后必须重启服务。

如使用其他 OpenAI-compatible 服务，可设置：

```powershell
$env:LLM_BASE_URL="https://你的服务/v1"
```

打开以下地址：

- 演示页面：`http://127.0.0.1:8000/`
- Swagger API：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## API 示例

先通过 `POST /api/v1/auth/register` 注册或通过 `POST /api/v1/auth/login`
登录获取 `access_token`，随后在分析请求头中加入
`Authorization: Bearer <access_token>`。`GET /api/v1/auth/me` 可读取当前用户。

```powershell
$body = @{ text = "You are a stupid loser." } | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/analyze `
  -ContentType application/json `
  -Body $body
```

没有设置 `LLM_API_KEY` 时，分类功能仍可用；关键词和提醒会明确标记为
`provider=fallback`，不会伪装成 LLM 结果。
