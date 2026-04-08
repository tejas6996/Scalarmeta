"""Phase 7 test: FastAPI server endpoints via TestClient."""
import json

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

# ============================================================
# TEST 1: Health check
# ============================================================
print("=" * 60)
print("TEST 1: Health check GET /")
print("=" * 60)

resp = client.get("/")
assert resp.status_code == 200
data = resp.json()
assert data["status"] == "ok"
assert "DisasterRelief" in data["environment"]
print(f"  {data}")
print("  PASS")

# ============================================================
# TEST 2: List tasks
# ============================================================
print()
print("=" * 60)
print("TEST 2: GET /tasks")
print("=" * 60)

resp = client.get("/tasks")
assert resp.status_code == 200
data = resp.json()
assert "tasks" in data
assert len(data["tasks"]) == 3
assert "task1_flood_easy" in data["tasks"]
print(f"  Tasks: {data['tasks']}")
print("  PASS")

# ============================================================
# TEST 3: Reset → Step → State → Grade flow
# ============================================================
print()
print("=" * 60)
print("TEST 3: Full POST /reset → /step → GET /state → POST /grade")
print("=" * 60)

# Reset
resp = client.post("/reset", json={"task_name": "task1_flood_easy", "seed": 42})
assert resp.status_code == 200
reset_data = resp.json()
assert "observation" in reset_data, "/reset must return {observation, reward, done, info}"
assert "reward" in reset_data
assert "done" in reset_data
assert reset_data["done"] is False
assert reset_data["reward"] == 0.0
obs = reset_data["observation"]
assert "step" in obs
assert "pending_reports" in obs
assert "available_resources" in obs
assert "zones" in obs
assert "available_tools" in obs
print(f"  Reset: step={obs['step']}, pending={len(obs['pending_reports'])}, "
      f"resources={len(obs['available_resources'])}, zones={len(obs['zones'])}")

# Step — intake first report
pending = obs["pending_reports"]
assert len(pending) > 0
rpt_id = pending[0]["id"]

resp = client.post("/step", json={"tool": "call_intake_agent", "args": {"report_id": rpt_id}})
assert resp.status_code == 200
result = resp.json()
assert "observation" in result
assert "reward" in result
assert "done" in result
assert "info" in result
print(f"  Step 1: reward={result['reward']:.2f}, done={result['done']}")

# Step — get_resources (no-op query)
resp = client.post("/step", json={"tool": "get_resources", "args": {}})
assert resp.status_code == 200
result = resp.json()
print(f"  Step 2: reward={result['reward']:.2f}, done={result['done']}")

# State
resp = client.get("/state")
assert resp.status_code == 200
state = resp.json()
assert "current_step" in state
assert "total_reward" in state
print(f"  State: step={state['current_step']}, reward={state['total_reward']:.1f}")

# Run remaining steps
while not result["done"]:
    resp = client.post("/step", json={"tool": "get_resources", "args": {}})
    assert resp.status_code == 200
    result = resp.json()

# Grade
resp = client.post("/grade")
assert resp.status_code == 200
grade = resp.json()
assert "score" in grade
assert "task_name" in grade
print(f"  Grade: score={grade['score']:.4f}, task={grade['task_name']}")
print("  PASS")

# ============================================================
# TEST 4: Error handling
# ============================================================
print()
print("=" * 60)
print("TEST 4: Error handling")
print("=" * 60)

# Step without reset after episode done — should still work (env allows it)
# but stepping on a done episode should give 400
resp = client.post("/step", json={"tool": "get_resources", "args": {}})
assert resp.status_code == 400
print(f"  Step after done: {resp.status_code} — {resp.json()['detail'][:50]}")

# Bad task name
resp = client.post("/reset", json={"task_name": "nonexistent_task"})
assert resp.status_code == 400
print(f"  Bad task: {resp.status_code} — {resp.json()['detail'][:50]}")

# Unknown tool after valid reset
resp = client.post("/reset", json={"task_name": "task1_flood_easy", "seed": 42})
assert resp.status_code == 200
resp = client.post("/step", json={"tool": "fly_to_moon", "args": {}})
assert resp.status_code == 200  # env handles gracefully, returns error in observation
result = resp.json()
assert result["observation"].get("last_action_error") is not None
print(f"  Unknown tool: handled gracefully, error in obs")

print("  PASS")

# ============================================================
# TEST 5: JSON serializable + content types
# ============================================================
print()
print("=" * 60)
print("TEST 5: All responses are valid JSON")
print("=" * 60)

resp = client.post("/reset", json={"task_name": "task2_storm_medium", "seed": 99})
obs_json = json.dumps(resp.json())
print(f"  /reset JSON: {len(obs_json)} bytes")

resp = client.post("/step", json={"tool": "get_resources", "args": {}})
step_json = json.dumps(resp.json())
print(f"  /step JSON: {len(step_json)} bytes")

resp = client.get("/state")
state_json = json.dumps(resp.json())
print(f"  /state JSON: {len(state_json)} bytes")

resp = client.post("/grade")
grade_json = json.dumps(resp.json())
print(f"  /grade JSON: {len(grade_json)} bytes")

print("  PASS")

# ============================================================
# TEST 6: Default reset (empty body)
# ============================================================
print()
print("=" * 60)
print("TEST 6: POST /reset with empty body (defaults)")
print("=" * 60)

resp = client.post("/reset", json={})
assert resp.status_code == 200
reset_data = resp.json()
obs = reset_data["observation"]
assert obs["task_name"] == "task1_flood_easy"
print(f"  Default reset: task={obs['task_name']}, step={obs['step']}")
print("  PASS")

print()
print("=== ALL PHASE 7 TESTS PASSED ===")
