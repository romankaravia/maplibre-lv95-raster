import Point from '@mapbox/point-geometry';
import {mat4, vec4} from 'gl-matrix';
import {MercatorTransform} from './mercator_transform.ts';
import {mercatorWorldCoordinateHelper} from '../mercator_coordinate.ts';
import {LngLatBounds} from '../lng_lat_bounds.ts';
import {calculateTileKey, type OverscaledTileID} from '../../tile/tile_id.ts';
import {Aabb} from '../../util/primitives/aabb.ts';
import {createMat4f64, type Mat4f32, type Mat4f64} from '../../util/util.ts';
import {EXTENT} from '../../data/extent.ts';
import {getMercatorHorizon, getMercatorTileCoordinates} from './mercator_utils.ts';
import {lv95WorldCoordinateHelper, lv95WorldToObliqueLongitude, mercatorToLV95World} from './lv95_utils.ts';
import {fitTilePolynomial, fitTilePolynomialAround, evaluateTilePolynomial, tileToLV95World, tileUnitsPerWorldUnit, type TileInGrid, type TilePolynomial} from './lv95_tile_reprojection.ts';
import {MercatorCoveringTilesDetailsProvider} from './mercator_covering_tiles_details_provider.ts';

import type {ITransform, NearFarZ} from '../transform_interface.ts';
import type {TransformOptions} from '../transform_helper.ts';
import type {UnwrappedTileID} from '../../tile/tile_id.ts';
import type {PointProjection} from '../../symbol/projection.ts';
import type {CustomLayerProjectionData, ProjectionDataParams, ProjectionReprojectionData, RendererProjectionData} from './projection_data.ts';
import type {CoveringTilesDetailsProvider} from './covering_tiles_details_provider.ts';
import type {CoveringTilesOptionsInternal} from './covering_tiles.ts';
import type {LngLat} from '../lng_lat.ts';
import type {IReadonlyTransform} from '../transform_interface.ts';

/**
 * The zoom level from which tiles reach the LV95 plane through polynomials fitted on the CPU instead of the exact
 * projection in the vertex shader. Below it a pixel spans 50 m or more, which the few meters of single precision in
 * world coordinates do not show; above it the polynomials keep tile positions precise to the deepest zoom.
 */
export const LV95_POLYNOMIAL_MIN_ZOOM = 10;

/** The largest distance in pixels the polynomial of a tile may put a position from its exact place. */
const POLYNOMIAL_TOLERANCE_PX = 0.05;

/** The most polynomials that interpolate whole tiles are kept, which only depend on the tile. */
const MAX_CACHED_TILE_POLYNOMIALS = 4096;

const REPROJECTION_PASSTHROUGH: ProjectionReprojectionData = {params: [0, 0, 0, 0], origin: [0, 0, 1, 0], coefficients: new Float32Array(16)};

/** How the tiles of one frame reach the LV95 plane and the screen. */
type LV95TileProjection = {
    reprojection: ProjectionReprojectionData;
    /** Takes planar tile units and an elevation in meters to clip space. */
    matrix: Mat4f64;
    matrix32: Mat4f32;
    /** Takes tile coordinates to planar tile units. */
    toPlane: (x: number, y: number) => [number, number];
    /** The linear approximation of {@link toPlane} at the tile center, computed when first needed. */
    linearization?: TileLinearization;
};

/**
 * `planar ≈ jacobian · tile + offset`, the jacobian column-major: the change of planar tile units per tile unit
 * along x, then along y.
 */
type TileLinearization = {
    jacobian: [number, number, number, number];
    offset: [number, number];
};

/**
 * The transform of the Swiss LV95 projection (EPSG:2056). The camera moves in the LV95 world plane, the Swiss grid
 * scaled to match Web Mercator at Bern, while tiles stay in the Web Mercator grid they are cut in and are reprojected
 * into the plane, by the vertex shader and, for symbols and queries, the same way on the CPU. Everything a camera does
 * in its world plane, the mercator transform already does for any {@link WorldCoordinateHelper}; this class adds how
 * tiles reach that plane.
 */
export class LV95Transform extends MercatorTransform {
    private readonly _lv95CoveringTilesDetailsProvider = new LV95CoveringTilesDetailsProvider();
    /** The projections of the tiles drawn this frame, which depend on the camera, by tile key. */
    private _tileProjections = new Map<string, LV95TileProjection>();
    /** Polynomials that interpolate whole tiles, which only depend on the tile, by tile key. */
    private _tilePolynomials = new Map<string, TilePolynomial>();
    private _fogMatrices = new Map<string, mat4>();

    constructor(options?: TransformOptions) {
        super({...options, worldCoordinateHelper: lv95WorldCoordinateHelper, tileWorldCoordinateHelper: mercatorWorldCoordinateHelper});
    }

    clone(): ITransform {
        const clone = new LV95Transform();
        clone.apply(this, false);
        return clone;
    }

    _calcMatrices(nearFarZ?: NearFarZ | null): void {
        super._calcMatrices(nearFarZ);
        this._tileProjections?.clear();
        this._fogMatrices?.clear();
    }

    getCoveringTilesDetailsProvider(): CoveringTilesDetailsProvider {
        return this._lv95CoveringTilesDetailsProvider;
    }

    getFastPathSimpleProjectionMatrix(_tileID: OverscaledTileID): mat4 {
        return null;
    }

    /**
     * Samples the edges of the viewport, which are curves in lng/lat: the bounds of the four corners alone miss the
     * parts of the view that bulge past them.
     */
    getBounds(): LngLatBounds {
        const top = Math.max(0, this.height / 2 - getMercatorHorizon(this));
        const bounds = new LngLatBounds();
        const steps = 16;
        for (let i = 0; i <= steps; i++) {
            const fractionX = this.width * i / steps;
            const fractionY = top + (this.height - top) * i / steps;
            bounds.extend(this.screenPointToLocation(new Point(fractionX, top)));
            bounds.extend(this.screenPointToLocation(new Point(fractionX, this.height)));
            bounds.extend(this.screenPointToLocation(new Point(0, fractionY)));
            bounds.extend(this.screenPointToLocation(new Point(this.width, fractionY)));
        }
        return bounds;
    }

    getProjectionData(params: ProjectionDataParams): RendererProjectionData {
        const {overscaledTileID, applyTerrainMatrix} = params;
        if (!overscaledTileID || (overscaledTileID.terrainRttPosMatrix32f && applyTerrainMatrix)) {
            return {...super.getProjectionData(params), reprojection: REPROJECTION_PASSTHROUGH};
        }
        const tileProjection = this._getTileProjection(overscaledTileID);
        return {
            mainMatrix: tileProjection.matrix32,
            tileMercatorCoords: getMercatorTileCoordinates(overscaledTileID),
            clippingPlane: [0, 0, 0, 0],
            projectionTransition: 0,
            fallbackMatrix: tileProjection.matrix32,
            clipAntimeridian: false,
            reprojection: tileProjection.reprojection,
        };
    }

    /**
     * Custom layers give Web Mercator positions in 0..1, which the shader prelude projects exactly, relative to the
     * center of the map; z is in world units, which match mercator's at Bern.
     */
    getProjectionDataForCustomLayer(_applyGlobeMatrix: boolean = true): CustomLayerProjectionData {
        const centerWorld = this.worldCoordinateHelper.worldFromLngLat(this.center.lng, this.center.lat);
        const matrix = createMat4f64();
        mat4.translate(matrix, this.modelViewProjectionMatrix, [centerWorld.x * this.worldSize, centerWorld.y * this.worldSize, 0]);
        mat4.scale(matrix, matrix, [this.worldSize, this.worldSize, this.worldSize / this.pixelsPerMeter]);
        return {
            mainMatrix: matrix,
            tileMercatorCoords: [0, 0, 1, 1],
            clippingPlane: [0, 0, 0, 0],
            projectionTransition: 0,
            fallbackMatrix: matrix,
            clipAntimeridian: false,
            reprojection: {params: [1, 0, 0, 0], origin: [centerWorld.x, centerWorld.y, 1, 0], coefficients: new Float32Array(16)},
        };
    }

    projectTileCoordinatesToPlane(x: number, y: number, tileID: UnwrappedTileID): Point {
        const [planarX, planarY] = this._getTileProjection(tileID).toPlane(x, y);
        return new Point(planarX, planarY);
    }

    projectPlanarTileCoordinates(x: number, y: number, tileID: UnwrappedTileID, elevation?: number): PointProjection {
        const position = vec4.transformMat4(new Float64Array(4), [x, y, elevation ?? 0, 1], this._getTileProjection(tileID).matrix);
        const w = position[3];
        return {
            point: new Point(position[0] / w, position[1] / w),
            signedDistanceFromCamera: w,
            isOccluded: false
        };
    }

    projectTileCoordinates(x: number, y: number, tileID: UnwrappedTileID, elevation?: number): PointProjection {
        const [planarX, planarY] = this._getTileProjection(tileID).toPlane(x, y);
        return this.projectPlanarTileCoordinates(planarX, planarY, tileID, elevation);
    }

    /** Inverts the linear approximation of a tile's reprojection at its center, see {@link getTileLinearization}. */
    tileCoordinatesFromPlane(x: number, y: number, tileID: UnwrappedTileID): Point {
        const {jacobian: [a, b, c, d], offset} = getTileLinearization(this._getTileProjection(tileID));
        const dx = x - offset[0];
        const dy = y - offset[1];
        const determinant = a * d - b * c;
        return new Point((d * dx - c * dy) / determinant, (a * dy - b * dx) / determinant);
    }

    /**
     * Fog depth only needs to follow the terrain smoothly, so each tile gets the linear approximation of its
     * reprojection at its center.
     */
    calculateFogMatrix(unwrappedTileID: UnwrappedTileID): mat4 {
        const key = unwrappedTileID.key;
        const cached = this._fogMatrices.get(key);
        if (cached) return cached;
        const tileProjection = this._getTileProjection(unwrappedTileID);
        const {jacobian, offset} = getTileLinearization(tileProjection);
        const linear = mat4.fromValues(
            jacobian[0], jacobian[1], 0, 0,
            jacobian[2], jacobian[3], 0, 0,
            0, 0, 1, 0,
            offset[0], offset[1], 0, 1);
        const unitsPerWorld = tileUnitsPerWorldUnit(unwrappedTileID.canonical.z);
        const [referenceX, referenceY] = tileProjection.reprojection.origin;
        const fogMatrix = createMat4f64();
        mat4.translate(fogMatrix, this.fogMatrix, [referenceX * this.worldSize, referenceY * this.worldSize, 0]);
        mat4.scale(fogMatrix, fogMatrix, [this.worldSize / unitsPerWorld, this.worldSize / unitsPerWorld, 1]);
        mat4.multiply(fogMatrix, fogMatrix, linear);
        const result = new Float32Array(fogMatrix);
        this._fogMatrices.set(key, result);
        return result;
    }

    /**
     * How a tile reaches the plane and the screen this frame. Below {@link LV95_POLYNOMIAL_MIN_ZOOM} the shader
     * projects it exactly. From there on a polynomial interpolates the whole tile where that is precise enough;
     * a tile too large for that, drawn far past its own zoom level, gets a polynomial around the point of the tile
     * nearest the center of the map.
     */
    private _getTileProjection(tileID: TileInGrid): LV95TileProjection {
        const {wrap, canonical} = tileID;
        const key = calculateTileKey(wrap, canonical.z, canonical.z, canonical.x, canonical.y);
        const cached = this._tileProjections.get(key);
        if (cached) return cached;

        const unitsPerWorld = tileUnitsPerWorldUnit(canonical.z);
        let tileProjection: LV95TileProjection;
        if (this.zoom < LV95_POLYNOMIAL_MIN_ZOOM) {
            const referenceWorld = tileToLV95World(tileID, EXTENT / 2, EXTENT / 2);
            tileProjection = this._createTileProjection(referenceWorld, unitsPerWorld, {params: [1, 0, 0, 0], origin: [referenceWorld[0], referenceWorld[1], 1, 0], coefficients: new Float32Array(16)}, (x, y) => {
                const [worldX, worldY] = tileToLV95World(tileID, x, y);
                return [(worldX - referenceWorld[0]) * unitsPerWorld, (worldY - referenceWorld[1]) * unitsPerWorld];
            });
        } else {
            const polynomial = this._getPolynomial(tileID, key, POLYNOMIAL_TOLERANCE_PX * unitsPerWorld / this.worldSize);
            tileProjection = this._createTileProjection(polynomial.referenceWorld, unitsPerWorld, {
                params: [2, polynomial.scale, polynomial.referenceX, polynomial.referenceY],
                origin: [polynomial.referenceWorld[0], polynomial.referenceWorld[1], polynomial.lineScale, 0],
                coefficients: new Float32Array(polynomial.coefficients),
            }, (x, y) => evaluateTilePolynomial(polynomial, x, y));
        }
        this._tileProjections.set(key, tileProjection);
        return tileProjection;
    }

    /** The polynomial for a tile whose error stays under `tolerance` planar tile units, see {@link _getTileProjection}. */
    private _getPolynomial(tileID: TileInGrid, key: string, tolerance: number): TilePolynomial {
        let wholeTile = this._tilePolynomials.get(key);
        if (!wholeTile) {
            if (this._tilePolynomials.size >= MAX_CACHED_TILE_POLYNOMIALS) this._tilePolynomials.clear();
            wholeTile = fitTilePolynomial(tileID, EXTENT / 2, EXTENT / 2, EXTENT / 2);
            this._tilePolynomials.set(key, wholeTile);
        }
        if (wholeTile.error <= tolerance) return wholeTile;
        const center = this.tileWorldCoordinateHelper.worldFromLngLat(this.center.lng, this.center.lat);
        const tilesAtZoom = Math.pow(2, tileID.canonical.z);
        const focusX = ((center.x - tileID.wrap) * tilesAtZoom - tileID.canonical.x) * EXTENT;
        const focusY = (center.y * tilesAtZoom - tileID.canonical.y) * EXTENT;
        return fitTilePolynomialAround(tileID, focusX, focusY, tolerance, wholeTile);
    }

    /**
     * The matrix from planar tile units to clip space is the view projection moved to the reference point and
     * scaled by the world size of a tile unit, computed in double precision like the tile matrices of mercator.
     */
    private _createTileProjection(referenceWorld: [number, number], unitsPerWorld: number, reprojection: ProjectionReprojectionData, toPlane: (x: number, y: number) => [number, number]): LV95TileProjection {
        const matrix = createMat4f64();
        mat4.translate(matrix, this.modelViewProjectionMatrix, [referenceWorld[0] * this.worldSize, referenceWorld[1] * this.worldSize, 0]);
        mat4.scale(matrix, matrix, [this.worldSize / unitsPerWorld, this.worldSize / unitsPerWorld, 1]);
        return {reprojection, matrix, matrix32: new Float32Array(matrix), toPlane};
    }
}

/** The linear approximation of a tile's reprojection at the tile center, by central differences. */
function getTileLinearization(tileProjection: LV95TileProjection): TileLinearization {
    if (tileProjection.linearization) return tileProjection.linearization;
    const center = EXTENT / 2;
    const step = EXTENT / 16;
    const east = tileProjection.toPlane(center + step, center);
    const west = tileProjection.toPlane(center - step, center);
    const south = tileProjection.toPlane(center, center + step);
    const north = tileProjection.toPlane(center, center - step);
    const [centerX, centerY] = tileProjection.toPlane(center, center);
    const jacobian: [number, number, number, number] = [
        (east[0] - west[0]) / (2 * step), (east[1] - west[1]) / (2 * step),
        (south[0] - north[0]) / (2 * step), (south[1] - north[1]) / (2 * step)
    ];
    tileProjection.linearization = {
        jacobian,
        offset: [centerX - (jacobian[0] + jacobian[2]) * center, centerY - (jacobian[1] + jacobian[3]) * center],
    };
    return tileProjection.linearization;
}

/** A bounding volume no camera ever sees, for tiles outside the world square. */
const OUTSIDE_WORLD = new Aabb([1e6, 1e6, 0], [1e6 + 1, 1e6 + 1, 0]);

/** The most world boxes of tiles that are kept, which only depend on the tile. */
const MAX_CACHED_TILE_BOXES = 16384;

/** An axis-aligned box in the LV95 world plane. */
type WorldBox = {minX: number; minY: number; maxX: number; maxY: number};

/**
 * Selects Web Mercator tiles by where they land in the LV95 plane: their bounding volumes are the boxes around
 * their reprojected outlines, and tiles outside the world square are left out.
 */
class LV95CoveringTilesDetailsProvider extends MercatorCoveringTilesDetailsProvider {
    private _worldBoxes = new Map<string, WorldBox | null>();

    getCenter(transform: IReadonlyTransform): LngLat {
        return transform.center;
    }

    getTileBoundingVolume(tileID: {x: number; y: number; z: number}, wrap: number, elevation: number, options: CoveringTilesOptionsInternal): Aabb {
        const box = this._getWorldBox(tileID);
        if (!box) return OUTSIDE_WORLD;
        const mercator = super.getTileBoundingVolume(tileID, wrap, elevation, options);
        return new Aabb([box.minX, box.minY, mercator.min[2]], [box.maxX, box.maxY, mercator.max[2]]);
    }

    allowWorldCopies(): boolean {
        return false;
    }

    private _getWorldBox(tileID: {x: number; y: number; z: number}): WorldBox | null {
        const key = `${tileID.z}/${tileID.x}/${tileID.y}`;
        if (this._worldBoxes.has(key)) return this._worldBoxes.get(key);
        if (this._worldBoxes.size >= MAX_CACHED_TILE_BOXES) this._worldBoxes.clear();
        const box = tileWorldBox(tileID);
        this._worldBoxes.set(key, box);
        return box;
    }
}

/** The oblique longitude past which a tile is taken to straddle the cut of the LV95 cylinder. */
const CUT_LONGITUDE = 2.5;

/**
 * The box around a tile's outline in the LV95 world plane, from a grid of samples grown by a margin for the bulge of
 * the curved edges between them, or null where it misses the world square. A tile that straddles the cut of the
 * cylinder lands on both sides of the plane: each side counts on its own, since the plane between them is not the tile.
 */
export function tileWorldBox(tileID: {x: number; y: number; z: number}): WorldBox | null {
    const tilesAtZoom = Math.pow(2, tileID.z);
    const steps = Math.max(4, 16 >> tileID.z);
    const sides: Array<WorldBox | null> = [null, null];
    for (let j = 0; j <= steps; j++) {
        for (let i = 0; i <= steps; i++) {
            const [x, y] = mercatorToLV95World((tileID.x + i / steps) / tilesAtZoom, (tileID.y + j / steps) / tilesAtZoom);
            const side = lv95WorldToObliqueLongitude(x) < -CUT_LONGITUDE ? 1 : 0;
            sides[side] ??= {minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity};
            const box = sides[side];
            box.minX = Math.min(box.minX, x);
            box.minY = Math.min(box.minY, y);
            box.maxX = Math.max(box.maxX, x);
            box.maxY = Math.max(box.maxY, y);
        }
    }
    let result: WorldBox | null = null;
    for (const side of sides) {
        if (!side) continue;
        const margin = 0.02 * Math.max(side.maxX - side.minX, side.maxY - side.minY);
        const grown = {minX: side.minX - margin, minY: side.minY - margin, maxX: side.maxX + margin, maxY: side.maxY + margin};
        if (grown.maxX < 0 || grown.minX > 1 || grown.maxY < 0 || grown.minY > 1) continue;
        result = result ? {
            minX: Math.min(result.minX, grown.minX), minY: Math.min(result.minY, grown.minY),
            maxX: Math.max(result.maxX, grown.maxX), maxY: Math.max(result.maxY, grown.maxY)
        } : grown;
    }
    return result;
}

/**
 * Creates a transform for the LV95 projection.
 */
export function createLV95Transform(options?: TransformOptions): LV95Transform {
    return new LV95Transform(options);
}
