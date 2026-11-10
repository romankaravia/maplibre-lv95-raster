import {EXTENT} from '../../data/extent.ts';
import {mercatorToLV95World} from './lv95_utils.ts';

/**
 * A tile in the Web Mercator grid: its canonical position and its world copy.
 */
export type TileInGrid = {
    wrap: number;
    canonical: {x: number; y: number; z: number};
};

/**
 * A biquadratic polynomial that takes tile coordinates near a reference point to the LV95 world plane, in planar tile
 * units: world units relative to the world position of the reference, divided by the world size of one tile unit.
 * With `u = (x - referenceX) * scale` and `v = (y - referenceY) * scale` it is
 * `u (c10 + u c20) + v (c01 + v c02) + u v (c11 + u c21 + v c12 + u v c22)`, the terms in `coefficients` as x/y
 * pairs in the order `c10, c01, c20, c11, c02, c21, c12, c22`, which is the column order of the shader's matrix.
 *
 * The vertex shader evaluates it in single precision: every term is small next to the planar tile units of the
 * tile's own extent, unlike world coordinates, whose seven significant digits are a few meters across the world.
 */
export type TilePolynomial = {
    referenceX: number;
    referenceY: number;
    /** The inverse of the half width, in tile units, of the 3 x 3 grid the polynomial interpolates. */
    scale: number;
    /** The world position of the reference point. */
    referenceWorld: [number, number];
    coefficients: Float64Array;
    /** The largest difference from the exact projection found between the grid points, in planar tile units. */
    error: number;
    /** How many planar tile units one tile unit spans at the reference point. */
    lineScale: number;
};

/** Weights of f(-1), f(0) and f(1) for the coefficients of 1, u and u² of the quadratic through the three values. */
const QUADRATIC_WEIGHTS = [[0, 1, 0], [-0.5, 0, 0.5], [0.5, -1, 0.5]] as const;

/** The powers of `u` and `v` of each term, in the order of {@link TilePolynomial.coefficients}. */
const TERMS = [[1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [2, 1], [1, 2], [2, 2]] as const;

/** Grid positions between the interpolated ones, where {@link TilePolynomial.error} is measured. */
const ERROR_SAMPLES = [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]] as const;

/** The number of tile units per world unit in a tile of the given zoom level: the inverse of a tile unit's world size. */
export function tileUnitsPerWorldUnit(z: number): number {
    return Math.pow(2, z) * EXTENT;
}

/** Projects a tile position to the LV95 world plane exactly. */
export function tileToLV95World(tile: TileInGrid, x: number, y: number): [number, number] {
    const tilesAtZoom = Math.pow(2, tile.canonical.z);
    return mercatorToLV95World(
        tile.wrap + (tile.canonical.x + x / EXTENT) / tilesAtZoom,
        (tile.canonical.y + y / EXTENT) / tilesAtZoom);
}

/**
 * Fits the polynomial that interpolates the exact projection on the 3 x 3 grid with the given half width in tile
 * units around a reference point of a tile, and measures its error between the grid points.
 */
export function fitTilePolynomial(tile: TileInGrid, referenceX: number, referenceY: number, halfWidth: number): TilePolynomial {
    const unitsPerWorld = tileUnitsPerWorldUnit(tile.canonical.z);
    const referenceWorld = tileToLV95World(tile, referenceX, referenceY);
    const planarAt = (u: number, v: number): [number, number] => {
        const [x, y] = tileToLV95World(tile, referenceX + u * halfWidth, referenceY + v * halfWidth);
        return [(x - referenceWorld[0]) * unitsPerWorld, (y - referenceWorld[1]) * unitsPerWorld];
    };
    const grid: Array<[number, number]> = [];
    for (let j = -1; j <= 1; j++) {
        for (let i = -1; i <= 1; i++) {
            grid.push(planarAt(i, j));
        }
    }
    const coefficients = new Float64Array(16);
    for (const [term, [powerU, powerV]] of TERMS.entries()) {
        for (let j = 0; j < 3; j++) {
            for (let i = 0; i < 3; i++) {
                const weight = QUADRATIC_WEIGHTS[powerU][i] * QUADRATIC_WEIGHTS[powerV][j];
                coefficients[2 * term] += weight * grid[3 * j + i][0];
                coefficients[2 * term + 1] += weight * grid[3 * j + i][1];
            }
        }
    }
    const polynomial: TilePolynomial = {
        referenceX,
        referenceY,
        scale: 1 / halfWidth,
        referenceWorld,
        coefficients,
        error: 0,
        lineScale: Math.hypot(coefficients[0], coefficients[1]) / halfWidth,
    };
    for (const [u, v] of ERROR_SAMPLES) {
        const exact = planarAt(u, v);
        const [x, y] = evaluateTilePolynomial(polynomial, referenceX + u * halfWidth, referenceY + v * halfWidth);
        polynomial.error = Math.max(polynomial.error, Math.hypot(exact[0] - x, exact[1] - y));
    }
    return polynomial;
}

/** Evaluates a {@link TilePolynomial} at a tile position, the way the vertex shader does. */
export function evaluateTilePolynomial(polynomial: TilePolynomial, x: number, y: number): [number, number] {
    const u = (x - polynomial.referenceX) * polynomial.scale;
    const v = (y - polynomial.referenceY) * polynomial.scale;
    const c = polynomial.coefficients;
    const uv = u * v;
    return [
        u * (c[0] + u * c[4]) + v * (c[2] + v * c[8]) + uv * (c[6] + u * c[10] + v * c[12] + uv * c[14]),
        u * (c[1] + u * c[5]) + v * (c[3] + v * c[9]) + uv * (c[7] + u * c[11] + v * c[13] + uv * c[15]),
    ];
}

/**
 * Fits a polynomial around the point of a tile nearest to `focusX`, `focusY`, in tile units, whose error stays under
 * `tolerance` planar tile units: the half width starts from the one that the error of `estimate`, which grows with
 * the cube of the half width, predicts, and is halved until the error is small enough. The reference point is snapped
 * to a power-of-two step well under the half width, so that it moves in exact steps as the focus moves and tile
 * positions minus the reference stay exact in single precision.
 */
export function fitTilePolynomialAround(tile: TileInGrid, focusX: number, focusY: number, tolerance: number, estimate: TilePolynomial): TilePolynomial {
    const estimateHalfWidth = 1 / estimate.scale;
    let halfWidth = estimate.error > 0 ? estimateHalfWidth * Math.cbrt(tolerance / estimate.error) * 0.8 : estimateHalfWidth;
    halfWidth = Math.pow(2, Math.floor(Math.log2(Math.min(halfWidth, estimateHalfWidth))));
    let polynomial: TilePolynomial;
    for (let attempt = 0; attempt < 12; attempt++) {
        const step = halfWidth / 16;
        const referenceX = Math.round(Math.min(Math.max(focusX, 0), EXTENT) / step) * step;
        const referenceY = Math.round(Math.min(Math.max(focusY, 0), EXTENT) / step) * step;
        polynomial = fitTilePolynomial(tile, referenceX, referenceY, halfWidth);
        if (polynomial.error <= tolerance) break;
        halfWidth /= 2;
    }
    return polynomial;
}
