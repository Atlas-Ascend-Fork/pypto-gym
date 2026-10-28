// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

export type GateFinding = {
  rule_id: string;
  severity: string;
  status: string;
  message: string;
  file: string;
};

export type GateSummary = {
  warnCount: number;
  infoCount: number;
  failFindings: GateFinding[];
};

export function formatGateFinding(finding: GateFinding): string {
  const messageAlreadyLocatesFile = finding.file !== "" &&
    finding.message.split("\n").some(
      (line) => line.startsWith(`${finding.file}:`),
    );
  const fileSuffix = finding.file && !messageAlreadyLocatesFile
    ? ` (${finding.file})`
    : "";
  return `  [${finding.rule_id}][${finding.severity}] ${finding.message}${fileSuffix}`;
}

type HookOutput = {
  hookSpecificOutput?: {
    additionalContext?: string;
    decision?: "allow" | "block";
    reason?: string;
  };
};

export type ParsedHookOutput = {
  additionalContext: string;
  decision: "allow" | "block";
  reason: string;
};

export function formatPluginError(scope: string, error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  return `[pypto-pro-op-lint plugin-error] ${scope} auto-check failed: ${detail}`;
}

export function parseHookOutput(raw: string): ParsedHookOutput {
  if (!raw.trim()) {
    return {
      additionalContext: "",
      decision: "block",
      reason: "[pypto-pro-op-lint plugin-error] post-edit auto-check returned empty output",
    };
  }

  try {
    const parsed = JSON.parse(raw) as HookOutput;
    const hookOutput = parsed.hookSpecificOutput;
    if (!hookOutput || (hookOutput.decision !== "allow" && hookOutput.decision !== "block")) {
      throw new Error("missing valid hookSpecificOutput.decision");
    }
    return {
      additionalContext: hookOutput.additionalContext ?? "",
      decision: hookOutput.decision,
      reason: hookOutput.reason ?? "",
    };
  } catch (error) {
    return {
      additionalContext: "",
      decision: "block",
      reason: formatPluginError("post-edit output parsing", error),
    };
  }
}

export function parseGateSummary(raw: string): GateSummary {
  if (!raw.trim()) {
    throw new Error("lint script returned empty output");
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const summary = parsed.summary;
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
      throw new Error("missing summary object");
    }
    const summaryRecord = summary as Record<string, unknown>;
    const countNames = ["pass", "warn", "info", "fail", "skip"] as const;
    const counts: Record<(typeof countNames)[number], number> = {
      pass: 0,
      warn: 0,
      info: 0,
      fail: 0,
      skip: 0,
    };
    for (const name of countNames) {
      const value = summaryRecord[name];
      if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
        throw new Error(`summary.${name} must be a non-negative integer`);
      }
      counts[name] = value;
    }
    if (typeof summaryRecord.has_error_fail !== "boolean") {
      throw new Error("summary.has_error_fail must be a boolean");
    }
    const findings = parsed.findings;
    if (!Array.isArray(findings)) {
      throw new Error("missing findings array");
    }
    const failFindings: GateFinding[] = [];
    const actualCounts = { pass: 0, warn: 0, info: 0, fail: 0, skip: 0 };
    for (const finding of findings) {
      if (!finding || typeof finding !== "object" || Array.isArray(finding)) {
        throw new Error("each finding must be an object");
      }
      const record = finding as Record<string, unknown>;
      const stringFields = ["rule_id", "severity", "status", "message", "file"];
      for (const field of stringFields) {
        if (typeof record[field] !== "string") {
          throw new Error(`finding.${field} must be a string`);
        }
      }
      if (!["S0", "S1", "S2", "S3"].includes(record.severity as string)) {
        throw new Error(`invalid finding severity: ${String(record.severity)}`);
      }
      const status = record.status as string;
      if (!["PASS", "WARN", "INFO", "FAIL", "SKIP"].includes(status)) {
        throw new Error(`invalid finding status: ${status}`);
      }
      const statusKey = status.toLowerCase() as keyof typeof actualCounts;
      actualCounts[statusKey] += 1;
      if (status === "FAIL") {
        failFindings.push({
          rule_id: record.rule_id as string,
          severity: record.severity as string,
          status: "FAIL",
          message: record.message as string,
          file: record.file as string,
        });
      }
    }
    for (const name of countNames) {
      if (actualCounts[name] !== counts[name]) {
        throw new Error(
          `summary.${name}=${counts[name]} does not match findings count ${actualCounts[name]}`,
        );
      }
    }
    const hasBlockingFailure = failFindings.some(
      (finding) => finding.severity === "S0" || finding.severity === "S1",
    );
    if (summaryRecord.has_error_fail !== hasBlockingFailure) {
      throw new Error("summary.has_error_fail does not match findings");
    }
    return { warnCount: counts.warn, infoCount: counts.info, failFindings };
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`invalid lint JSON output: ${detail}`);
  }
}
