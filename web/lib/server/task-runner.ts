import { spawn } from "node:child_process";

export type BufferedProcessResult = {
  code: number | null;
  stdout: string;
  stderr: string;
};

const MAX_LOG_CHARS = 16000;

export function appendChunk(current: string, chunk: string): string {
  const next = current + chunk;
  if (next.length <= MAX_LOG_CHARS) {
    return next;
  }
  return next.slice(next.length - MAX_LOG_CHARS);
}

export function trimLog(log: string): string {
  return log.trim().slice(-6000);
}

export function getGlobalState<T>(key: string, initialValue: T): T {
  const record = globalThis as Record<string, unknown>;
  if (!(key in record)) {
    record[key] = initialValue;
  }
  return record[key] as T;
}

export function runBufferedProcess(command: string, args: string[], options: { cwd: string; env?: NodeJS.ProcessEnv }): Promise<BufferedProcessResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env || process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk: Buffer | string) => {
      stdout = appendChunk(stdout, chunk.toString());
    });
    child.stderr.on("data", (chunk: Buffer | string) => {
      stderr = appendChunk(stderr, chunk.toString());
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}

export function wireLineBufferedStream(stream: NodeJS.ReadableStream, onLine: (line: string) => void): void {
  let buffer = "";
  stream.on("data", (chunk: Buffer | string) => {
    buffer += chunk.toString();
    while (buffer.includes("\n")) {
      const newlineIndex = buffer.indexOf("\n");
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        onLine(line);
      }
    }
  });
  stream.on("end", () => {
    const remainder = buffer.trim();
    if (remainder) {
      onLine(remainder);
    }
  });
}
