import http from "k6/http";
import { check } from "k6";
import { Counter } from "k6/metrics";
import { SharedArray } from "k6/data";
import exec from "k6/execution";

const payloads = new SharedArray("payloads", function () {
  return JSON.parse(open("./payloads.json"));
});

const BASE_URL = __ENV.BASE_URL || "http://fraud-api:8000";
const VUS = Number(__ENV.VUS || 1);
const DURATION = __ENV.DURATION || "60s";
const WARMUP_DURATION = __ENV.WARMUP_DURATION || "15s";
const ACTIONS = ["approve", "step_up", "review", "decline"];

const semanticFailures = new Counter("semantic_failures");
const actionCounters = {};
for (const name of ACTIONS) {
  actionCounters[name] = new Counter("action_" + name);
}

function durationSeconds(value) {
  const match = String(value).trim().match(/^(\d+)(ms|s|m|h)$/);
  if (!match) {
    throw new Error("bad duration: " + value);
  }
  const amount = Number(match[1]);
  const scale = { ms: 0.001, s: 1, m: 60, h: 3600 };
  return amount * scale[match[2]];
}

export const options = {
  summaryTrendStats: ["avg", "min", "med", "p(95)", "p(99)", "max"],
  scenarios: {
    warmup: {
      executor: "constant-vus",
      vus: 1,
      duration: WARMUP_DURATION,
      tags: { phase: "warmup" },
    },
    measured: {
      executor: "constant-vus",
      vus: VUS,
      duration: DURATION,
      startTime: durationSeconds(WARMUP_DURATION) + 5 + "s",
      tags: { phase: "measured" },
    },
  },
  thresholds: {
    "http_req_duration{phase:measured}": ["max>=0"],
    "http_reqs{phase:measured}": ["count>=0"],
    "http_req_failed{phase:measured}": ["rate>=0"],
    "semantic_failures{phase:measured}": ["count>=0"],
    "action_approve{phase:measured}": ["count>=0"],
    "action_step_up{phase:measured}": ["count>=0"],
    "action_review{phase:measured}": ["count>=0"],
    "action_decline{phase:measured}": ["count>=0"],
  },
};

let primed = false;

function primeMeasuredSeries() {
  if (primed) {
    return;
  }
  primed = true;
  semanticFailures.add(0, { phase: "measured" });
  for (const name of ACTIONS) {
    actionCounters[name].add(0, { phase: "measured" });
  }
}

export default function () {
  const phase = exec.scenario.name;
  if (phase === "measured") {
    primeMeasuredSeries();
  }
  const item = payloads[(__VU * 7919 + __ITER) % payloads.length];
  const response = http.post(
    BASE_URL + "/predict",
    JSON.stringify(item.request),
    { headers: { "Content-Type": "application/json" } }
  );
  let data = null;
  if (response.status === 200) {
    try {
      data = response.json();
    } catch (_error) {
      data = null;
    }
  }
  const ok = check(response, {
    "status is 200": function () {
      return response.status === 200;
    },
    "has fraud_probability": function () {
      return data !== null && typeof data.fraud_probability === "number";
    },
    "has recommended_action": function () {
      return data !== null && typeof data.recommended_action === "string";
    },
    "has model_version": function () {
      return data !== null && typeof data.model_version === "string";
    },
    "has request_id": function () {
      return data !== null && typeof data.request_id === "string";
    },
    "probability in range": function () {
      return (
        data !== null &&
        data.fraud_probability >= 0 &&
        data.fraud_probability <= 1
      );
    },
    "action known": function () {
      return data !== null && ACTIONS.indexOf(data.recommended_action) !== -1;
    },
    "model is v9": function () {
      return data !== null && data.model_version === "v9";
    },
  });
  if (response.status === 200 && !ok) {
    semanticFailures.add(1, { phase: phase });
  }
  if (ok && data !== null) {
    actionCounters[data.recommended_action].add(1, { phase: phase });
  }
}

function metricValue(data, name, key) {
  const metric = data.metrics[name];
  if (!metric || !metric.values || metric.values[key] === undefined) {
    return 0;
  }
  return metric.values[key];
}

export function handleSummary(data) {
  const durationS = durationSeconds(DURATION);
  const requests = metricValue(data, "http_reqs{phase:measured}", "count");
  const summary = {
    vus: VUS,
    duration_s: durationS,
    requests: requests,
    achieved_rps: durationS === 0 ? 0 : requests / durationS,
    http_failures: metricValue(data, "http_req_failed{phase:measured}", "passes"),
    semantic_failures: metricValue(
      data,
      "semantic_failures{phase:measured}",
      "count"
    ),
    p50_ms: metricValue(data, "http_req_duration{phase:measured}", "med"),
    p95_ms: metricValue(data, "http_req_duration{phase:measured}", "p(95)"),
    p99_ms: metricValue(data, "http_req_duration{phase:measured}", "p(99)"),
    max_ms: metricValue(data, "http_req_duration{phase:measured}", "max"),
    actions: {
      approve: metricValue(data, "action_approve{phase:measured}", "count"),
      step_up: metricValue(data, "action_step_up{phase:measured}", "count"),
      review: metricValue(data, "action_review{phase:measured}", "count"),
      decline: metricValue(data, "action_decline{phase:measured}", "count"),
    },
  };
  console.log("K6_SUMMARY " + JSON.stringify(summary));
  return {};
}
