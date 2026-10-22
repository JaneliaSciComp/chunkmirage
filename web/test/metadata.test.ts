// worldarray's shared fixtures (test/fixtures/worldarray/, its tests/fixtures as released in
// 0.3.0, the version chunkmirage pins): every metadata layout placed by the browser's readers
// (src/placement.ts) where worldarray places it, and every OME multiscales document the
// browser writes (src/layout.ts) as worldarray writes it. Run by `npm test`; when the pin
// moves, copy the two files from that release.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { attrs2, group3 } from "../src/layout.ts";
import { axisTypes, placement, placeN5, placePrecomputed, placeZarr, type Attrs } from "../src/placement.ts";
import type { ViewInfo } from "../src/types.ts";

const fixture = (name: string): Attrs[] => JSON.parse(readFileSync(new URL(`fixtures/worldarray/${name}`, import.meta.url), "utf8")).cases;

/** A case's placement, from its documents as worldarray's tests write them (conftest.py's
 * build_case): an N5 array's attributes beside its dimensions, the group's beside "n5". */
function place(c: Attrs) {
  const n = c.shape.length, name = c.array ?? "s0";
  if (c.format === "precomputed") return placement(placePrecomputed(c as { resolution: number[] }), n);
  if (c.format === "n5") return placement(placeN5({ dimensions: [...c.shape].reverse(), ...c.attrs }, { n5: "2.5.0", ...c.group_attrs }, name, n), n);
  return placement(placeZarr(c.attrs ?? {}, c.group_attrs ?? {}, name, n, c.dimension_names), n);
}

const close = (got: number[], want: number[], what: string) =>
  assert.ok(got.length === want.length && got.every((v, i) => Math.abs(v - want[i]) < 1e-9), `${what} ${got} is not ${want}`);

for (const c of fixture("metadata.json")) {
  test(`placed as worldarray places it: ${c.name}`, () => {
    const p = place(c), e = c.expect;
    close(p.voxel, e.voxel_size, "voxel size");
    close(p.origin, e.translation, "translation");
    assert.deepEqual([p.axes, p.named, p.units, p.placedBy], [e.axes, e.axes_named, e.units, e.placed_by]);
  });
}

/** A view of the case's axes, in its resolved units, and its levels, each voxel size and
 * translation over every axis (given over the space axes alone, the others 1 and 0). */
function viewOf(c: Attrs): ViewInfo {
  const { axis_units: units, axis_types: types } = c.resolved as { axis_units: string[]; axis_types: (string | null)[] };
  const space = types.flatMap((t, i) => (t === "space" ? [i] : []));
  const every = (v: number[], fill: number) => (v.length === types.length ? v : types.map((_, i) => (space.includes(i) ? v[space.indexOf(i)] : fill)));
  return {
    dtype: "uint8", channels: 1, halo: [], lead: 0, out: "uint8", added: [],
    axes: (c.axes as string[]).map((name, i) => ({ name, unit: units[i] })),
    levels: (c.levels as [string, number[], number[]][]).map(([, voxel, origin]) => ({ shape: [], voxel: every(voxel, 1), origin: every(origin, 0) })),
  };
}

// the browser writes neither a multiscale's type nor its metadata, so the case giving them is left out
for (const c of fixture("write.json").filter((c) => !c.multiscale_type && !c.metadata)) {
  test(`OME ${c.version} written as worldarray writes it: ${c.name}`, () => {
    const v = viewOf(c), datasets = (c.levels as [string][]).map(([path], level) => ({ path, level }));
    const got = c.version === "0.4" ? attrs2("", v, datasets) : group3("", v, datasets).attributes;
    assert.deepEqual(JSON.parse(JSON.stringify(got)), c.expect);
  });
}

test("an array with no attributes of its own is placed by its group, its axes typed for OME", () => {
  const group = { multiscales: [{ axes: ["t", "c", "z", "y", "x"], datasets: [{ path: "0", coordinateTransformations: [{ type: "scale", scale: [1, 1, 4, 2, 2] }] }] }] };
  const p = placement(placeZarr(undefined, group, "0", 5), 5);  // zarrita gives an array without .zattrs no attrs
  assert.deepEqual([p.voxel, p.placedBy], [[1, 1, 4, 2, 2], "ome"]);
  assert.deepEqual(axisTypes(p.axes, p.units), ["time", "channel", "space", "space", "space"]);  // the register page writes them
});
