import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchExperiments } from "./api";

function apiResponse() {
  return new Response(JSON.stringify({ data: [] }), {
    headers: { "Content-Type": "application/json" },
  });
}

describe("fetchExperiments", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shares a pending request, then fetches again after it settles", async () => {
    let resolveFirst: ((value: Response) => void) | undefined;
    const firstResponse = new Promise<Response>((resolve) => {
      resolveFirst = resolve;
    });
    const fetchMock = vi.fn().mockReturnValueOnce(firstResponse).mockResolvedValueOnce(apiResponse());
    vi.stubGlobal("fetch", fetchMock);

    const first = fetchExperiments(true);
    const duplicateStrictModeRequest = fetchExperiments(true);
    expect(duplicateStrictModeRequest).toBe(first);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    resolveFirst?.(apiResponse());
    await expect(first).resolves.toEqual([]);
    await Promise.resolve();

    await expect(fetchExperiments(true)).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not coalesce requests with different summary variants", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL) => Promise.resolve(apiResponse()));
    vi.stubGlobal("fetch", fetchMock);

    await Promise.all([fetchExperiments(false), fetchExperiments(true)]);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/experiments?with_summary=false",
      "/api/experiments?with_summary=true",
    ]);
  });
});
