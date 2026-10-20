// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { expect, mock, spyOn, test } from "bun:test";
import { $ } from "bun";
import fs, { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const optionalSchema = () => ({ optional: optionalSchema });
const tool = Object.assign(
  <T>(definition: T): T => definition,
  {
    schema: {
      number: optionalSchema,
      string: optionalSchema,
    },
  },
);

mock.module("@opencode-ai/plugin", () => ({ tool }));

const { PyptoStateTransitionPlugin, resolveSkillScript } = await import("../pypto-state-transition");

type StateTransitionTool = {
  execute: (
    args: Record<string, unknown>,
    context: Record<string, unknown>,
  ) => Promise<string>;
};

async function makeTool(directory: string, worktree = directory): Promise<StateTransitionTool> {
  const plugin = await PyptoStateTransitionPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory,
    worktree,
    project: {},
  } as never);
  return plugin.tool?.state_transition as unknown as StateTransitionTool;
}

test("harness generic executor/verifier agents may drive state_transition", async () => {
  for (const agent of ["general-executor", "general-verifier"]) {
    const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-harness-"));
    try {
      const transition = await makeTool(worktree);
      await transition.execute(
        {
          action: "init",
          stage: 1,
          max_stage: 7,
          opDir: "custom/smoke_op",
        },
        { agent },
      );
      const statePath = join(worktree, "custom", "smoke_op", ".orchestrator_state.json");
      expect(existsSync(statePath)).toBeTrue();
    } finally {
      rmSync(worktree, { recursive: true, force: true });
    }
  }
});

test("named classic subagents are still rejected from state_transition", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-deny-"));
  try {
    const transition = await makeTool(worktree);
    for (const agent of ["pypto-op-coder", "pypto-op-verifier", ""]) {
      await expect(
        transition.execute(
          {
            action: "init",
            stage: 1,
            max_stage: 7,
            opDir: "custom/smoke_op",
          },
          { agent },
        ),
      ).rejects.toThrow("permission denied");
      expect(existsSync(join(worktree, "custom", "smoke_op"))).toBeFalse();
    }
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init creates a missing operator directory under custom", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-"));
  try {
    const transition = await makeTool(worktree);
    await transition.execute(
      {
        action: "init",
        stage: 1,
        max_stage: 7,
        opDir: "custom/smoke_op",
      },
      { agent: "build" },
    );

    const statePath = join(worktree, "custom", "smoke_op", ".orchestrator_state.json");
    expect(existsSync(statePath)).toBeTrue();
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    expect(state.operator_name).toBe("smoke_op");
    expect(state.stage_status["1"]).toBe("in_progress");
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init rejects an output directory outside custom before creating it", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-"));
  try {
    const transition = await makeTool(worktree);
    await expect(
      transition.execute(
        {
          action: "init",
          stage: 1,
          max_stage: 7,
          opDir: "other/smoke_op",
        },
        { agent: "build" },
      ),
    ).rejects.toThrow("may only update an operator directory");
    expect(existsSync(join(worktree, "other", "smoke_op"))).toBeFalse();
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("init resolves custom under the active directory rather than a broader worktree", async () => {
  const worktree = mkdtempSync(join(tmpdir(), "pypto-state-transition-worktree-"));
  const directory = join(worktree, "project");
  mkdirSync(directory);
  try {
    const transition = await makeTool(directory, worktree);
    await transition.execute(
      {
        action: "init",
        stage: 1,
        max_stage: 7,
        opDir: "custom/smoke_op",
      },
      { agent: "build" },
    );

    expect(existsSync(join(directory, "custom", "smoke_op", ".orchestrator_state.json"))).toBeTrue();
    expect(existsSync(join(worktree, "custom", "smoke_op", ".orchestrator_state.json"))).toBeFalse();
  } finally {
    rmSync(worktree, { recursive: true, force: true });
  }
});

test("design output failure blocks progress and retry checks both outputs again", async () => {
  const directory = mkdtempSync(join(tmpdir(), "pypto-design-handoff-"));
  // The repo checkout has no assembled skills dir; pretend the design
  // checker script exists so the gate proceeds to the mocked shell.
  const realExistsSync = fs.existsSync;
  const existsSpy = spyOn(fs, "existsSync").mockImplementation((p: unknown) =>
    String(p).includes("validate_artifacts.py") ? true : realExistsSync(p as fs.PathLike),
  );
  let complete = false;
  const calls: string[] = [];
  const shell = (strings: TemplateStringsArray, ...values: unknown[]) => {
    const command = strings.reduce((s, part, i) => s + part + String(values[i] ?? ""), "");
    calls.push(command);
    const result = {
      cwd: () => result,
      quiet: () => result,
      nothrow: async () => ({
        exitCode: command.includes("validate_artifacts.py") && !complete ? 1 : 0,
        stdout: Buffer.from(command.includes("validate_artifacts.py") ? "interface check" : '{}'),
        stderr: Buffer.from(""),
      }),
    };
    return result;
  };
  try {
    const initial = await makeTool(directory);
    await initial.execute({action: "init", stage: 1, max_stage: 7, opDir: "custom/demo"}, {agent: "build"});
    const statePath = join(directory, "custom/demo/.orchestrator_state.json");
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    state.current_stage = 3;
    state.stage_status["1"] = state.stage_status["2"] = "completed";
    state.stage_status["3"] = "in_progress";
    writeFileSync(statePath, JSON.stringify(state));
    const plugin = await PyptoStateTransitionPlugin({ $: shell, directory, worktree: directory,
      client: {app: {log: async () => {}}}, project: {} } as never);
    const transition = plugin.tool?.state_transition as unknown as StateTransitionTool;
    await expect(transition.execute({action: "complete_stage", stage: 3, opDir: "custom/demo"}, {agent: "build"}))
      .rejects.toThrow("pypto-op-architect");
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(3);
    complete = true;
    await transition.execute({action: "complete_stage", stage: 3, opDir: "custom/demo"}, {agent: "build"});
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(4);
    await transition.execute({action: "submit_design", stage: 4, opDir: "custom/demo"}, {agent: "build"});
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(4);
    expect(calls.filter(c => c.includes("validate_artifacts.py"))).toHaveLength(3);
  } finally {
    existsSpy.mockRestore();
    rmSync(directory, {recursive: true, force: true});
  }
});

// ─── resolveSkillScript: skill script resolution across workspace layouts ───
// In both assembled layouts this hook lands at `<workspace>/.opencode/plugins/`.
// The gym/opencode layout assembles skills into `.opencode/skills/`; the
// cannbot installer assembles them into `.agents/skills/`.

const SCRIPT_REL = "pypto-op-design/scripts/validate_artifacts.py";

function makeLayout(kind: "gym" | "cannbot" | "both" | "none") {
  const ws = mkdtempSync(join(tmpdir(), `pypto-skill-script-${kind}-`));
  const hookUrl = pathToFileURL(
    join(ws, ".opencode", "plugins", "pypto-state-transition.ts"),
  ).href;
  if (kind === "gym" || kind === "both") {
    const scriptPath = join(ws, ".opencode", "skills", SCRIPT_REL);
    mkdirSync(dirname(scriptPath), { recursive: true });
    writeFileSync(scriptPath, "# stub\n");
  }
  if (kind === "cannbot" || kind === "both") {
    const scriptPath = join(ws, ".agents", "skills", SCRIPT_REL);
    mkdirSync(dirname(scriptPath), { recursive: true });
    writeFileSync(scriptPath, "# stub\n");
  }
  return { ws, hookUrl };
}

test("resolveSkillScript finds the script in the gym layout (.opencode/skills)", () => {
  const { ws, hookUrl } = makeLayout("gym");
  try {
    expect(resolveSkillScript(SCRIPT_REL, hookUrl)).toBe(
      join(ws, ".opencode", "skills", SCRIPT_REL),
    );
  } finally {
    rmSync(ws, { recursive: true, force: true });
  }
});

test("resolveSkillScript falls back to the cannbot layout (.agents/skills)", () => {
  const { ws, hookUrl } = makeLayout("cannbot");
  try {
    expect(resolveSkillScript(SCRIPT_REL, hookUrl)).toBe(
      join(ws, ".agents", "skills", SCRIPT_REL),
    );
  } finally {
    rmSync(ws, { recursive: true, force: true });
  }
});

test("resolveSkillScript prefers the gym layout when both exist", () => {
  const { ws, hookUrl } = makeLayout("both");
  try {
    expect(resolveSkillScript(SCRIPT_REL, hookUrl)).toBe(
      join(ws, ".opencode", "skills", SCRIPT_REL),
    );
  } finally {
    rmSync(ws, { recursive: true, force: true });
  }
});

test("resolveSkillScript reports every candidate path when the script is missing", () => {
  const { ws, hookUrl } = makeLayout("none");
  try {
    let message = "";
    try {
      resolveSkillScript(SCRIPT_REL, hookUrl);
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    expect(message).toContain(`skill script not found: ${SCRIPT_REL}`);
    expect(message).toContain(join(ws, ".opencode", "skills", SCRIPT_REL));
    expect(message).toContain(join(ws, ".agents", "skills", SCRIPT_REL));
  } finally {
    rmSync(ws, { recursive: true, force: true });
  }
});

test("design gate rejects with a clear error when the checker script is not assembled", async () => {
  // This repo checkout has no assembled skills dir next to the hook, so the
  // real resolver hits the missing-script branch end to end.
  const directory = mkdtempSync(join(tmpdir(), "pypto-design-gate-missing-"));
  const calls: string[] = [];
  const shell = (strings: TemplateStringsArray, ...values: unknown[]) => {
    const command = strings.reduce((s, part, i) => s + part + String(values[i] ?? ""), "");
    calls.push(command);
    const result = {
      cwd: () => result,
      quiet: () => result,
      nothrow: async () => ({
        exitCode: 0,
        stdout: Buffer.from("{}"),
        stderr: Buffer.from(""),
      }),
    };
    return result;
  };
  try {
    const initial = await makeTool(directory);
    await initial.execute(
      { action: "init", stage: 1, max_stage: 7, opDir: "custom/demo" },
      { agent: "build" },
    );
    const statePath = join(directory, "custom/demo/.orchestrator_state.json");
    const state = JSON.parse(readFileSync(statePath, "utf8"));
    state.current_stage = 3;
    state.stage_status["1"] = state.stage_status["2"] = "completed";
    state.stage_status["3"] = "in_progress";
    writeFileSync(statePath, JSON.stringify(state));
    const plugin = await PyptoStateTransitionPlugin({
      $: shell,
      directory,
      worktree: directory,
      client: { app: { log: async () => {} } },
      project: {},
    } as never);
    const transition = plugin.tool?.state_transition as unknown as StateTransitionTool;
    await expect(
      transition.execute({ action: "complete_stage", stage: 3, opDir: "custom/demo" }, { agent: "build" }),
    ).rejects.toThrow("skill script not found");
    // The gate must fail before any shell command runs, and the state file
    // must stay untouched.
    expect(calls).toHaveLength(0);
    expect(JSON.parse(readFileSync(statePath, "utf8")).current_stage).toBe(3);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
