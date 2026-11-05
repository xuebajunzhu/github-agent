# Security Policy

## Reporting a vulnerability

请通过 GitHub 私有安全公告报告（仓库 → Security → Report a vulnerability），
**不要**在公开 issue 中描述可利用细节。确认后 72 小时内响应，修复或缓解后公开披露。

本项目是个人维护的开源项目，无专职安全团队；重要修复随 main 分支发布并在
CHANGELOG.md 标注。

## 部署安全模型

| 面 | 现状 |
| --- | --- |
| 管理端点 `/api/admin/*` | `X-Admin-Token` 头 + 常数时间比较；生产必须换掉默认值（启动时告警） |
| 公开端点 | `/api/recommend`、`/api/research`（每 IP 5/分 + 30/时 预算）、`/api/voice/*`（10MB 上传上限）、`/api/feedback`、`/mcp`（可选 `MCP_AUTH_TOKEN` Bearer） |
| 限流 | 全局每客户端滑窗 + 研究端点独立预算；隧道部署需 `TRUST_PROXY_HEADERS=true`（仅信任回环来源转发头） |
| 响应头 | CSP（限定 self）/ nosniff / DENY / no-referrer 中间件 |
| 凭据存储 | GitHub OAuth token Fernet 加密（密钥派生自 SECRET_KEY）；`.env` 不入库；CI 每次推送跑凭据模式扫描 |
| LLM 供应链 | 仓库文本是**不可信输入**：分析输出强制严格 JSON schema；模型加载 `weights_only=True`；Release 解包 `filter="data"` |

## 已接受风险（知情取舍，勿重复报告）

1. **git 历史含旧数据库产物**（2026-11-04 前的 WAL/训练数据）——tip 已清除，
   历史清洗需 BFG+force push，已在路线图中；
2. **OAuth/JWT 链路是半成品**——签发后无路由消费（死代码即攻击面，接线或删除
   在路线图中）；
3. **LLM 提示词注入面**：仓库描述/README 由第三方控制并进入分析提示词；输出被
   严格 JSON schema 约束、前端全量 HTML 转义、索引文本仅影响检索排序——影响
   评估为低，缓解依赖金标准门禁 + summary judge；
4. **单机 SQLite + 文件锁**——多机部署需启用 REDIS_URL 并迁移 PostgreSQL
   （compose 已备）；
5. **`MCP_AUTH_TOKEN` 默认未设置**——MCP 端点公开可读（受全局限流），公网长期
   部署建议设置；
6. **actions 按 tag 引用**——dependabot 周更 + 权限最小化缓解，SHA 固定在路线图中。

## 安全相关配置速查

`SECRET_KEY` / `ADMIN_TOKEN`（必换，启动告警）/ `MCP_AUTH_TOKEN` / `TRUST_PROXY_HEADERS`
/ `REDIS_URL` / `FERNET_KEY` —— 见 `.env.example` 注释。
