# MCP 目录提交材料（推广就绪包）

本文件是 `github-agent` MCP server 提交到各目录的**就绪材料**。提交前置条件
与操作步骤见文末。

## 1. awesome-mcp-servers 条目（wong2/awesome-mcp-servers 格式）

分类建议：`🔍 Data & Analysis`（或该仓库现行分类惯例）。

```markdown
- **[github-agent](https://github.com/xuebajunzhu/github-agent)** —
  场景驱动的开源项目推荐与调研：输入自然语言问题（中英皆可），返回直接可用
  方案 + 可替代方案，每条带实时验证信号（stars/license/维护度/最近推送）。
  索引每 10 分钟自愈巡检、每日自动扩召；数据来自实时抓取，不依赖模型记忆，
  杜绝幻觉包名。`research_problem` 工具可做选型对比与生态调研。
```

提交方式：fork → 分支 → 按字母序插入 → PR。**前置条件见第 4 节。**

## 2. 官方 MCP Registry 元数据（registry.modelcontextprotocol.io）

```json
{
  "name": "github-agent",
  "display_name": "GitHub Agent — verified open-source project finder",
  "description": "Scenario-driven recommendations and problem research over a live, continuously refreshed index of GitHub open-source projects. Returns stars, license, maintenance tier and last-push for every result — grounded in real data, immune to hallucinated package names.",
  "tools": [
    {"name": "recommend_scenario", "description": "Semantic scenario -> verified projects (top_k)"},
    {"name": "research_problem", "description": "Problem -> direct solutions + substitutable alternatives, with per-item justifications"},
    {"name": "list_projects", "description": "Browse the index by language/stars with maintenance signals"},
    {"name": "get_project", "description": "Look up one indexed repository by owner/repo"}
  ],
  "transports": ["streamable-http"],
  "auth": "optional bearer token (MCP_AUTH_TOKEN)"
}
```

## 3. 客户端连接指南（给使用者的说明）

### Claude Code / Cursor（mcp-remote 桥接，适合自建部署）

```json
{
  "mcpServers": {
    "github-agent": {
      "command": "npx",
      "args": ["mcp-remote", "https://<你的域名或隧道地址>/mcp",
               "--header", "Authorization: Bearer ${MCP_AUTH_TOKEN}"]
    }
  }
}
```

### 裸 HTTP（任何支持 Streamable HTTP 的客户端）

```
POST https://<host>/mcp
Content-Type: application/json
Authorization: Bearer <MCP_AUTH_TOKEN>   # 未设置 MCP_AUTH_TOKEN 时可省略

{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

## 4. 提交前置条件（按顺序）

1. **命名 Cloudflare 隧道**（当前快速隧道 URL 随重启变化，目录拒收动态地址）：
   `cloudflared tunnel login` → `cloudflared tunnel create github-agent` →
   config.yml 绑定 `http://localhost:8788` → `cloudflared tunnel run github-agent`
   → 固定域名 `<name>.<account>.trycloudflare.com` 或自有域。
2. **设置 `MCP_AUTH_TOKEN`**（公网服务必须）：`.env` 一行 + `scripts\manage.ps1 restart`。
3. **用固定域名做一轮端到端验证**：`POST /mcp` initialize → tools/list → 一次
   `recommend_scenario` 调用（脚本 `scripts/integration_test.py` 可复用）。
4. 按 1-2 节材料提交 PR / registry 注册；README 的 MCP 段落同步更新固定域名。

## 5. 目录之外的推广面（零外部依赖，已完成）

- `skills/choose-dependency/SKILL.md` — anthropics/skills 格式的选型技能，
  任何 skills 兼容 agent 可直接安装使用（仓库内分发，无外部审批）。
- README「MCP Server」段落 + 连接示例已就绪。
- 每次推送的 CI 已含 lint/secrets/test 三门禁，提交前可本地复跑：
  `ruff check ... && python scripts/scan_secrets.py && pytest -q`。
