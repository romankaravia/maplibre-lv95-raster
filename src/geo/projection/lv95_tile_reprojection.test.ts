import {describe, expect, test} from 'vitest';
import {EXTENT} from '../../data/extent.ts';
import {evaluateTilePolynomial, fitTilePolynomial, fitTilePolynomialAround, tileToLV95World, tileUnitsPerWorldUnit} from './lv95_tile_reprojection.ts';

/** The z14 tile over Bern. */
const bernTile = {wrap: 0, canonical: {z: 14, x: 8530, y: 5766}};

/** The exact planar tile units of a tile position relative to the world position of a reference point. */
function exactPlanar(tile: typeof bernTile, referenceWorld: [number, number], x: number, y: number): [number, number] {
    const [worldX, worldY] = tileToLV95World(tile, x, y);
    const unitsPerWorld = tileUnitsPerWorldUnit(tile.canonical.z);
    return [(worldX - referenceWorld[0]) * unitsPerWorld, (worldY - referenceWorld[1]) * unitsPerWorld];
}

describe('LV95 tile polynomials', () => {
    test('interpolate the exact projection on their grid and stay far under a pixel between its points', () => {
        const polynomial = fitTilePolynomial(bernTile, EXTENT / 2, EXTENT / 2, EXTENT / 2);
        for (const [x, y] of [[0, 0], [EXTENT, 0], [EXTENT / 2, EXTENT], [EXTENT, EXTENT]]) {
            const [px, py] = evaluateTilePolynomial(polynomial, x, y);
            const [ex, ey] = exactPlanar(bernTile, polynomial.referenceWorld, x, y);
            expect(Math.hypot(px - ex, py - ey)).toBeLessThan(1e-6);
        }
        const [px, py] = evaluateTilePolynomial(polynomial, 1000, 7000);
        const [ex, ey] = exactPlanar(bernTile, polynomial.referenceWorld, 1000, 7000);
        expect(Math.hypot(px - ex, py - ey)).toBeLessThan(1e-3);
        expect(polynomial.error).toBeLessThan(1e-3);
    });

    test('scale tile units by the ratio of the LV95 and Web Mercator scales, which is 1 at Bern', () => {
        const polynomial = fitTilePolynomial(bernTile, EXTENT / 2, EXTENT / 2, EXTENT / 2);
        expect(polynomial.lineScale).toBeCloseTo(1, 3);
        const hamburgTile = {wrap: 0, canonical: {z: 14, x: 8646, y: 5295}};
        const hamburg = fitTilePolynomial(hamburgTile, EXTENT / 2, EXTENT / 2, EXTENT / 2);
        const mercatorScaleRatio = Math.cos(53.55 * Math.PI / 180) / Math.cos(46.95 * Math.PI / 180);
        expect(Math.abs(hamburg.lineScale / mercatorScaleRatio - 1)).toBeLessThan(0.01);
    });

    test('narrow around a focus point until a tile drawn far past its zoom level meets the tolerance', () => {
        const largeTile = {wrap: 0, canonical: {z: 6, x: 33, y: 22}};
        const wholeTile = fitTilePolynomial(largeTile, EXTENT / 2, EXTENT / 2, EXTENT / 2);
        const tolerance = 1e-4;
        expect(wholeTile.error).toBeGreaterThan(tolerance);

        const focus = 1234.567;
        const polynomial = fitTilePolynomialAround(largeTile, focus, focus, tolerance, wholeTile);
        expect(polynomial.error).toBeLessThanOrEqual(tolerance);
        const step = 1 / polynomial.scale / 16;
        expect(Number.isInteger(polynomial.referenceX / step)).toBe(true);
        expect(Math.abs(polynomial.referenceX - focus)).toBeLessThanOrEqual(step / 2);
        const [px, py] = evaluateTilePolynomial(polynomial, focus + 0.3 / polynomial.scale, focus);
        const [ex, ey] = exactPlanar(largeTile, polynomial.referenceWorld, focus + 0.3 / polynomial.scale, focus);
        expect(Math.hypot(px - ex, py - ey)).toBeLessThanOrEqual(tolerance);
    });
});
