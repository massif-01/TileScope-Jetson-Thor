set -euxo pipefail
base=/opt/tilescope
candidate="$1"
trial="$2"
e="$base/evidence"
repo="$base/$trial"
py=/opt/tilescope/venv/bin/python
ev="$e/$trial"
mkdir "$ev"
git clone -b feat/multimodel-probe "$base/$candidate.bundle" "$repo"
cd "$repo"
test "$(git rev-parse HEAD)" = "$candidate"
python3 -m tools.multimodel.config_challenge before --repo "$repo" --evidence "$ev"
cp "$base/config-only/profiles/models/qwen3-small.json" profiles/models/
cp "$base/config-only/profiles/targets/T2.json" profiles/targets/
git status --porcelain=v1 --untracked-files=all > "$ev/added.txt"
git diff --no-index /dev/null profiles/models/qwen3-small.json > "$ev/model.diff" || test "$?" = 1
git diff --no-index /dev/null profiles/targets/T2.json > "$ev/target.diff" || test "$?" = 1
target=profiles/targets/T2.json
"$py" -m tilescope.probe.owned --target "$target" --out "$ev/owner" --depth "$e/T2-depth-$trial" > "$ev/owner.log" 2>&1 &
owner=$!
cleanup(){ touch "$ev/owner/STOP"; wait "$owner"; }
trap cleanup EXIT
until test -f "$ev/owner/ready.json"; do kill -0 "$owner"; sleep 2; done
"$py" -m tilescope probe check --target "$target" --suite workloads/suites/new-target.json --out "$ev/check.json"
for s in protocol-pilot chat-pair new-target same-content concurrency; do
  "$py" -m tilescope probe run --target "$target" --suite "workloads/suites/$s.json" --out "$e/T2-$s-$trial"
done
"$py" -m tilescope probe run --target "$target" --suite workloads/suites/deep-dense.json --out "$e/T2-depth-$trial"
"$py" -m tilescope.probe.depth checkpoint-review --out "$e/T2-depth-$trial" --checkpoint "$base/models/Qwen3-0.6B" > "$ev/checkpoint-review.json"
python3 -m tools.multimodel.config_challenge after --repo "$repo" --evidence "$ev" --allow profiles/models/qwen3-small.json profiles/targets/T2.json
