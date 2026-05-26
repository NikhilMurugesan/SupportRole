# Active Interview Context

Use this context for live interview answers until this file is replaced for a new interview.

Interview: Avathon Forward Deployed AI Engineer / SCM resource-allocation assignment.

Candidate project: Resource Allocation Engine for a delivery fleet.

Primary framing:
- This is an operational decisioning system, not a generic GenAI chatbot.
- The project assigns trucks to delivery orders under hard physical constraints.
- The domain language is trucks, orders/parcels, routes, cost matrix, Greedy, Hungarian, cheapest insertion, capacity, capabilities, time windows, SLA, shift end, distance, and workload balance.

Architecture facts:
- Backend: FastAPI, SQLite, SQLAlchemy, Pydantic.
- Frontend: React, Vite, Leaflet/OpenStreetMap, SVG view.
- Shared logic: Haversine distance, travel-time conversion, cheapest-insertion router, soft scoring.
- Algorithms: Greedy allocation and Hungarian batch allocation using SciPy `linear_sum_assignment`.
- Both algorithms share the same feasibility rules and scoring so comparison is fair.

Answering rules:
- If the interviewer says "assignment", "submission", "problem", "algorithm", "router.py", "route class", "cost matrix", "row/column", "truck", "parcel/order", or "best insertion", answer in the Resource Allocation Engine context.
- Do not answer with UPS Supervisor Assistant, GTS, Bedrock, API Gateway, XGBoost, LOF, BERT, RAG, or generic ML unless the interviewer explicitly asks for those.
- For vague fragments, infer the current topic from the latest concrete Avathon question. If no real question is present, wait.
- Be honest about limitations: local demo, small dataset, Haversine not road distance, Hungarian is one-to-one per round and not full VRP.

Important prepared answers:
- Greedy: sort orders by priority and SLA, try every truck, run cheapest insertion, score feasible candidates, commit the best truck, mark infeasible orders unassigned.
- Hungarian: build an order-by-truck cost matrix, use `linear_sum_assignment` to pick the minimum total cost set of one-to-one assignments for that round, commit feasible pairs, update routes, repeat.
- Location enters before the solver: truck/order latitude and longitude are converted to distance/travel time during cheapest insertion, and that score becomes the cost matrix value.
- `linear_sum_assignment` returns row and column indices. In this project, those indices point back to selected order-truck candidate pairs, then the code commits each chosen insertion into the matching truck route.
- Best insertion complexity is roughly O(R^2) per order-truck candidate because it tries R + 1 insertion positions and recomputes route timing across stops. Greedy is approximately O(N * M * R^2), plus sorting. Hungarian matrix construction is also O(N_remaining * M * R^2) per round, plus solver cost.
- Scaling 10,000 orders and 1,000 trucks requires candidate pruning, geography partitioning, cached road-distance matrices, async/background jobs, parallel matrix construction, incremental reoptimization, and stronger solvers like OR-Tools VRPTW, min-cost flow, MILP, or CP-SAT.
- Distance computation is a pure repeated function call, so cache/precompute a distance matrix for repeated truck/order/depot points. This is not the same as dynamic programming over changing state.
- AI-first Avathon answer: start from the customer operational problem, model resources/demands/constraints, use optimization for guaranteed feasibility, and use agents/LLMs for workflow orchestration, explanation, policy retrieval, planner assistance, validation, and fast shipping.
