# 饲料配方优化服务（feedmill）

后端服务：版本化管理原料库与配方规格，用**自实现的两阶段单纯形**做最低成本
配方优化，给出可逐条代入检验的最优性自证、影子价格、检验数、价格灵敏度区间
和不可行冲突诊断；支持换价后的异步批量重优化作业。

* Python 3.12 + FastAPI + NumPy（**不使用任何 LP/优化库**）
* SQLite 持久化，数据落在挂载卷 `/data`
* 只提供 HTTP 接口，无页面

## 快速开始

```bash
docker compose up --build        # 服务在 http://localhost:8000
# 或本地
pip install -r requirements.txt
FEED_DATA_DIR=./data uvicorn app.main:app --reload
```

运行测试：

```bash
pip install pytest httpx
pytest -q
```

## 典型流程（每周一换价）

```bash
# 1) 维护原料（营养含量用小数：0.08=8%），发布为新版本
curl -s -XPOST localhost:8000/api/library/publish -H 'Content-Type: application/json' -d '{
  "message": "第40周报价",
  "ingredients": [
    {"code":"corn","name":"玉米","price":0.30,"nutrients":{"DM":0.88,"CP":0.08}},
    {"code":"sbm","name":"豆粕","price":0.50,"nutrients":{"DM":0.89,"CP":0.44}}
  ]}'

# 2) 新建/修改配方规格（修改产生新版本）
curl -s -XPUT localhost:8000/api/recipes/layer1 -H 'Content-Type: application/json' -d '{
  "code":"layer1","name":"蛋鸡料","total_mass":1.0,
  "ingredients":[{"code":"corn","min_ratio":0,"max_ratio":1},
                 {"code":"sbm","min_ratio":0,"max_ratio":1}],
  "nutrients":[{"code":"CP","lower":0.18}],
  "ratios":[]}'

# 3) 即时优化（含自证、影子价格、检验数、灵敏度）
curl -s -XPOST localhost:8000/api/optimize -H 'Content-Type: application/json' \
  -d '{"recipe_code":"layer1"}'

# 4) 换价后批量重优化所有受影响配方（立即返回作业号）
curl -s -XPOST localhost:8000/api/jobs -H 'Content-Type: application/json' -d '{}'
curl -s localhost:8000/api/jobs/1                 # 查进度
curl -s -XPOST localhost:8000/api/jobs/1/cancel  # 取消

# 5) 历史与版本对比
curl -s 'localhost:8000/api/recipes/layer1/results'
curl -s -XPOST localhost:8000/api/compare -H 'Content-Type: application/json' \
  -d '{"recipe_code":"layer1","result_a":1,"result_b":2}'
```

## 手算核对算例

玉米 CP 8% @0.30 元/kg、豆粕 CP 44% @0.50 元/kg，要求 CP ≥ 18%：

```
豆粕 = (0.18-0.08)/(0.44-0.08) = 0.2778 kg
玉米 = 0.7222 kg，成本 = 0.3556 元
```

`POST /api/optimize` 返回 `amounts_kg={corn:0.72222, sbm:0.27778}`、
`cost=0.355556`，CP 约束起作用、影子价格 0.5556，三条件全部通过、
原始/对偶目标相对差 ~1e-16。

## 主要接口

| 方法 路径 | 说明 |
|---|---|
| `GET /api/nutrients` / `POST /api/nutrients` | 营养项（可扩展；内置 DM/CP/ME/CA/TP/LYS/MET） |
| `POST /api/library/publish` | 以全量快照发布原料库新版本 |
| `GET /api/library?version=` / `GET /api/library/versions` / `GET /api/library/diff` | 版本查询与差异 |
| `PUT /api/recipes/{code}` | 新建或修改配方（产生新版本） |
| `GET /api/recipes/{code}?version=` / `.../versions` | 配方版本 |
| `POST /api/optimize` | 即时优化（可指定库/配方版本、`warm_start`） |
| `GET /api/recipes/{code}/results` / `GET /api/results/{id}` | 结果历史 / 详情 |
| `POST /api/compare` | 同一配方两个结果的用量差、成本差 |
| `POST /api/jobs` / `GET /api/jobs/{id}` / `POST /api/jobs/{id}/cancel` | 批量作业（提交时锁定库与每个配方的规格版本，详情条目带 `recipe_version`） |

## 优化结果报告字段（节选）

* `amounts_kg`：各原料用量；`cost`：总成本；`achieved_nutrients`：达成营养值
* `constraints[]`：每条约束的 lhs/rhs/松弛/是否起作用/对偶价格/可行性与互补松弛
* `reduced_costs[]`：每种原料的检验数
* `certificate`：`primal_feasible / dual_feasible / complementary_slackness /
  relative_gap / optimal_certificate_ok`
* `price_sensitivity`：每种原料保持当前组成不变的价格区间及挡住区间的约束
* `binding_constraints` / `shadow_prices`
* 不可行时：`status=infeasible` + `diagnosis`（可读冲突说明 + 极小冲突约束子集）
* `warm_start`：是否尝试/命中热启动、来源库版本、是否回退冷启动
* `recipe_version` / `library_version`：结果绑定的两个版本

求解器、对偶/灵敏度、不可行诊断与热启动取舍的细节见 [DESIGN.md](DESIGN.md)。
