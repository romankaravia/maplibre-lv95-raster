import {LngLat} from '../lng_lat.ts';
import {MercatorCoordinate, latFromMercatorY, lngFromMercatorX, mercatorXfromLng, mercatorYfromLat} from '../mercator_coordinate.ts';
import {clamp} from '../../util/util.ts';

import type {WorldCoordinateHelper} from '../transform_interface.ts';
import type {TileMatrixSet} from './tile_matrix_set.ts';

/** Semi-major axis of the WGS84 ellipsoid, in meters. */
const WGS84_A = 6378137;
/** First eccentricity squared of the WGS84 ellipsoid, from its flattening of 1 / 298.257223563. */
const WGS84_E2 = 0.00669437999014;
/** Semi-major axis of the Bessel 1841 ellipsoid, which CH1903+ and LV95 are built on, in meters. */
const BESSEL_A = 6377397.155;
/** First eccentricity squared of the Bessel 1841 ellipsoid. */
const BESSEL_E2 = 0.006674372230614;
const BESSEL_E = Math.sqrt(BESSEL_E2);

/**
 * The geocentric translation from CH1903+ to WGS84 in meters, the three-parameter datum transformation swisstopo
 * publishes for LV95. It is good to about a decimeter across Switzerland.
 */
const CH1903PLUS_TO_WGS84 = [674.374, 15.056, 405.346] as const;

/** Latitude of the projection center, the old observatory of Bern, on the Bessel ellipsoid: 46°57'08.66". */
const PHI0 = (46 + 57 / 60 + 8.66 / 3600) * Math.PI / 180;
/** Longitude of the projection center: 7°26'22.50". */
const LAMBDA0 = (7 + 26 / 60 + 22.5 / 3600) * Math.PI / 180;

/** Radius of the projection sphere, the Gauss sphere tangent to the Bessel ellipsoid at the projection center. */
const LV95_R = BESSEL_A * Math.sqrt(1 - BESSEL_E2) / (1 - BESSEL_E2 * Math.sin(PHI0) ** 2);
/** Ratio between longitudes on the projection sphere and on the ellipsoid. */
const LV95_ALPHA = Math.sqrt(1 + BESSEL_E2 / (1 - BESSEL_E2) * Math.cos(PHI0) ** 4);
/** Latitude of the projection center on the projection sphere. */
const LV95_B0 = Math.asin(Math.sin(PHI0) / LV95_ALPHA);
/** Integration constant of the conformal mapping from the ellipsoid to the projection sphere. */
const LV95_K = Math.atanh(Math.sin(LV95_B0)) - LV95_ALPHA * Math.atanh(Math.sin(PHI0)) + LV95_ALPHA * BESSEL_E * Math.atanh(BESSEL_E * Math.sin(PHI0));
const SIN_B0 = Math.sin(LV95_B0);
const COS_B0 = Math.cos(LV95_B0);

/** LV95 easting of the projection center, in meters. */
export const LV95_EASTING_ORIGIN = 2600000;
/** LV95 northing of the projection center, in meters. */
export const LV95_NORTHING_ORIGIN = 1200000;

/**
 * The constants of the Swiss oblique conformal cylindrical projection, as swisstopo defines them in
 * "Formulas and constants for the calculation of the Swiss conformal cylindrical projection and for the
 * transformation between coordinate systems". The vertex shader prelude of the projection repeats them.
 */
export const LV95_PROJECTION_CONSTANTS: Readonly<Record<'R' | 'alpha' | 'b0' | 'K' | 'lambda0' | 'besselE', number>> = {
    R: LV95_R,
    alpha: LV95_ALPHA,
    b0: LV95_B0,
    K: LV95_K,
    lambda0: LAMBDA0,
    besselE: BESSEL_E,
};

/**
 * Converts a WGS84 latitude and longitude in radians, on the ellipsoid, to CH1903+ on the Bessel ellipsoid through
 * geocentric coordinates and swisstopo's three-parameter datum shift. The ellipsoidal height is taken as zero; it
 * moves the result by about a decimeter per kilometer of height.
 */
function wgs84ToCH1903Plus(lambda: number, phi: number): [number, number] {
    const [x, y, z] = geocentric(lambda, phi, 0, WGS84_A, WGS84_E2);
    const [lambdaB, phiB] = geodetic(x - CH1903PLUS_TO_WGS84[0], y - CH1903PLUS_TO_WGS84[1], z - CH1903PLUS_TO_WGS84[2], BESSEL_A, BESSEL_E2);
    return [lambdaB, phiB];
}

/**
 * The inverse of {@link wgs84ToCH1903Plus}. A point at zero WGS84 height lies off the Bessel ellipsoid, so the
 * CH1903+ height it had is solved for, which makes the round trip exact.
 */
function ch1903PlusToWgs84(lambda: number, phi: number): [number, number] {
    let heightB = 0;
    let result: [number, number, number];
    for (let i = 0; i < 3; i++) {
        const [x, y, z] = geocentric(lambda, phi, heightB, BESSEL_A, BESSEL_E2);
        result = geodetic(x + CH1903PLUS_TO_WGS84[0], y + CH1903PLUS_TO_WGS84[1], z + CH1903PLUS_TO_WGS84[2], WGS84_A, WGS84_E2);
        heightB -= result[2];
    }
    return [result[0], result[1]];
}

/** Geocentric coordinates in meters of a longitude and latitude in radians at a height in meters above an ellipsoid. */
function geocentric(lambda: number, phi: number, height: number, a: number, e2: number): [number, number, number] {
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const n = a / Math.sqrt(1 - e2 * sinPhi * sinPhi);
    return [(n + height) * cosPhi * Math.cos(lambda), (n + height) * cosPhi * Math.sin(lambda), (n * (1 - e2) + height) * sinPhi];
}

/**
 * The longitude and latitude in radians and the height in meters of a geocentric position on an ellipsoid, by
 * fixed-point iteration on the height. Points near the surface converge to well under a millimeter in four passes.
 */
function geodetic(x: number, y: number, z: number, a: number, e2: number): [number, number, number] {
    const p = Math.hypot(x, y);
    let phi = Math.atan2(z, p * (1 - e2));
    let height = 0;
    for (let i = 0; i < 5; i++) {
        const sinPhi = Math.sin(phi);
        const n = a / Math.sqrt(1 - e2 * sinPhi * sinPhi);
        height = Math.abs(Math.cos(phi)) > 1e-9 ? p / Math.cos(phi) - n : Math.abs(z) - n * (1 - e2);
        phi = Math.atan2(z, p * (1 - e2 * n / (n + height)));
    }
    return [Math.atan2(y, x), phi, height];
}

/**
 * Projects a CH1903+ longitude and latitude in radians to LV95 easting and northing in meters relative to the
 * projection center: from the ellipsoid to the projection sphere, then onto the oblique cylinder through Bern.
 * The oblique longitude is taken in -π..π, so the cut of the cylinder lies opposite Bern on the globe.
 */
function ch1903PlusToLV95Offset(lambda: number, phi: number): [number, number] {
    const s = LV95_ALPHA * Math.atanh(Math.sin(phi)) - LV95_ALPHA * BESSEL_E * Math.atanh(BESSEL_E * Math.sin(phi)) + LV95_K;
    const sinB = Math.tanh(s);
    const cosB = 1 / Math.cosh(s);
    const l = LV95_ALPHA * (lambda - LAMBDA0);
    const sinObliqueB = clamp(COS_B0 * sinB - SIN_B0 * cosB * Math.cos(l), -1, 1);
    const obliqueL = Math.atan2(cosB * Math.sin(l), SIN_B0 * sinB + COS_B0 * cosB * Math.cos(l));
    return [LV95_R * obliqueL, LV95_R * Math.atanh(sinObliqueB)];
}

/** The inverse of {@link ch1903PlusToLV95Offset}, iterating on the latitude of the ellipsoid. */
function lv95OffsetToCH1903Plus(east: number, north: number): [number, number] {
    const obliqueL = east / LV95_R;
    const sinObliqueB = Math.tanh(north / LV95_R);
    const cosObliqueB = 1 / Math.cosh(north / LV95_R);
    const sinB = clamp(COS_B0 * sinObliqueB + SIN_B0 * cosObliqueB * Math.cos(obliqueL), -1, 1);
    const l = Math.atan2(cosObliqueB * Math.sin(obliqueL), COS_B0 * cosObliqueB * Math.cos(obliqueL) - SIN_B0 * sinObliqueB);
    const lambda = LAMBDA0 + l / LV95_ALPHA;
    if (Math.abs(sinB) >= 1) return [lambda, Math.sign(sinB) * Math.PI / 2];
    const psi = (Math.atanh(sinB) - LV95_K) / LV95_ALPHA;
    let phi = Math.asin(sinB);
    for (let i = 0; i < 8; i++) {
        const next = 2 * Math.atan(Math.exp(psi + BESSEL_E * Math.atanh(BESSEL_E * Math.sin(phi)))) - Math.PI / 2;
        const done = Math.abs(next - phi) < 1e-14;
        phi = next;
        if (done) break;
    }
    return [lambda, phi];
}

/**
 * Converts a WGS84 location in degrees to LV95 (EPSG:2056) easting and northing in meters, with swisstopo's
 * rigorous projection formulas and its three-parameter datum shift.
 */
export function lngLatToLV95(lng: number, lat: number): [number, number] {
    const [lambda, phi] = wgs84ToCH1903Plus(lng * Math.PI / 180, lat * Math.PI / 180);
    const [east, north] = ch1903PlusToLV95Offset(lambda, phi);
    return [LV95_EASTING_ORIGIN + east, LV95_NORTHING_ORIGIN + north];
}

/** Converts LV95 (EPSG:2056) easting and northing in meters to a WGS84 location in degrees, the inverse of {@link lngLatToLV95}. */
export function lv95ToLngLat(east: number, north: number): [number, number] {
    const [lambda, phi] = lv95OffsetToCH1903Plus(east - LV95_EASTING_ORIGIN, north - LV95_NORTHING_ORIGIN);
    const [lng, lat] = ch1903PlusToWgs84(lambda, phi);
    return [lng * 180 / Math.PI, lat * 180 / Math.PI];
}

/**
 * The WGS84 location of the LV95 projection center, which the LV95 world plane puts where Web Mercator puts it.
 */
const ORIGIN_LNG_LAT = lv95ToLngLat(LV95_EASTING_ORIGIN, LV95_NORTHING_ORIGIN);

/**
 * World units per LV95 meter: the Web Mercator scale at the projection center, measured along the easting, so that a
 * zoom level shows Bern at the same size in both projections and a map style keeps its look around Switzerland.
 */
const WORLD_UNITS_PER_METER = (() => {
    const step = 1;
    const [westLng] = lv95ToLngLat(LV95_EASTING_ORIGIN - step, LV95_NORTHING_ORIGIN);
    const [eastLng] = lv95ToLngLat(LV95_EASTING_ORIGIN + step, LV95_NORTHING_ORIGIN);
    return (mercatorXfromLng(eastLng) - mercatorXfromLng(westLng)) / (2 * step);
})();

/**
 * How the LV95 grid sits in the world square the camera moves in: easting grows along x and northing against y,
 * scaled by {@link WORLD_UNITS_PER_METER}, with the projection center at its Web Mercator position. Near Bern the
 * world coordinates of LV95 and of Web Mercator therefore nearly agree; they drift apart by about a meter 10 km away
 * and a kilometer 300 km away.
 */
export const LV95_WORLD: Readonly<Record<'originX' | 'originY' | 'unitsPerMeter', number>> = {
    originX: mercatorXfromLng(ORIGIN_LNG_LAT[0]),
    originY: mercatorYfromLat(ORIGIN_LNG_LAT[1]),
    unitsPerMeter: WORLD_UNITS_PER_METER,
};

/** Converts LV95 easting and northing in meters to world coordinates. */
export function lv95ToWorld(east: number, north: number): [number, number] {
    return [
        LV95_WORLD.originX + (east - LV95_EASTING_ORIGIN) * LV95_WORLD.unitsPerMeter,
        LV95_WORLD.originY - (north - LV95_NORTHING_ORIGIN) * LV95_WORLD.unitsPerMeter
    ];
}

/** Converts world coordinates to LV95 easting and northing in meters. */
export function worldToLV95(x: number, y: number): [number, number] {
    return [
        LV95_EASTING_ORIGIN + (x - LV95_WORLD.originX) / LV95_WORLD.unitsPerMeter,
        LV95_NORTHING_ORIGIN - (y - LV95_WORLD.originY) / LV95_WORLD.unitsPerMeter
    ];
}

/**
 * Projects a Web Mercator position, in the 0..1 square tiles are cut in, to LV95 world coordinates. The latitude of
 * `y` past the mercator square, as the pole vertices of a tile mesh use, is clamped to the poles.
 */
export function mercatorToLV95World(x: number, y: number): [number, number] {
    const lat = y <= 0 ? 90 : y >= 1 ? -90 : latFromMercatorY(y);
    const [lambda, phi] = wgs84ToCH1903Plus(lngFromMercatorX(x) * Math.PI / 180, lat * Math.PI / 180);
    const [east, north] = ch1903PlusToLV95Offset(lambda, phi);
    return [LV95_WORLD.originX + east * LV95_WORLD.unitsPerMeter, LV95_WORLD.originY - north * LV95_WORLD.unitsPerMeter];
}

/**
 * The oblique longitude in radians, -π..π, at which a position with the given LV95 world x lies on the cylinder.
 * It is 0 on the oblique meridian through Bern and jumps from π to -π across the cut of the cylinder, opposite Bern
 * on the globe.
 */
export function lv95WorldToObliqueLongitude(x: number): number {
    return (x - LV95_WORLD.originX) / LV95_WORLD.unitsPerMeter / LV95_R;
}

/** Converts LV95 world coordinates to a Web Mercator position in the 0..1 square tiles are cut in. */
export function lv95WorldToMercator(x: number, y: number): [number, number] {
    const [lambda, phi] = lv95OffsetToCH1903Plus((x - LV95_WORLD.originX) / LV95_WORLD.unitsPerMeter, (LV95_WORLD.originY - y) / LV95_WORLD.unitsPerMeter);
    const [lng, lat] = ch1903PlusToWgs84(lambda, phi);
    return [mercatorXfromLng(lng * 180 / Math.PI), mercatorYfromLat(lat * 180 / Math.PI)];
}

/**
 * The world square of the LV95 projection in LV95 meters: tile 0/0/0 of the quad grid over the world plane.
 */
const LV95_WORLD_TILE_MATRIX_SET: TileMatrixSet = {
    origin: worldToLV95(0, 0),
    extentAtZoom0: 1 / LV95_WORLD.unitsPerMeter,
};

/**
 * @internal
 * The world plane of the LV95 projection, which the camera of an LV95 map moves in: the Swiss grid, scaled and placed
 * as {@link LV95_WORLD} describes. The plane is flat and bounded by its 0..1 world square, so it neither wraps nor
 * renders world copies. Tiles are still cut in Web Mercator; the projection reprojects them into this plane.
 */
class LV95WorldCoordinateHelper implements WorldCoordinateHelper {
    readonly name = 'lv95';
    readonly wraps = false;
    readonly tileMatrixSet = LV95_WORLD_TILE_MATRIX_SET;

    worldFromLngLat(lng: number, lat: number, altitude?: number): MercatorCoordinate {
        const [east, north] = lngLatToLV95(lng, clamp(lat, -90, 90));
        const [x, y] = lv95ToWorld(east, north);
        return new MercatorCoordinate(x, y, altitude === undefined ? 0 : altitude * LV95_WORLD.unitsPerMeter);
    }

    lngLatFromWorld(x: number, y: number): LngLat {
        const [east, north] = worldToLV95(x, y);
        const [lng, lat] = lv95ToLngLat(east, north);
        return new LngLat(lng, clamp(lat, -90, 90));
    }

    /** Positions inside the world square, the part of the plane the map renders. */
    hasLngLat(x: number, y: number): boolean {
        return x >= 0 && x <= 1 && y >= 0 && y <= 1;
    }

    metersPerWorldUnit(_x: number, _y: number): number {
        return 1 / LV95_WORLD.unitsPerMeter;
    }

    worldZFromAltitude(altitude: number, _lngLat: LngLat): number {
        return altitude * LV95_WORLD.unitsPerMeter;
    }
}

/**
 * @internal
 * The one LV95 world plane: it holds no state, so every LV95 transform shares this instance.
 */
export const lv95WorldCoordinateHelper: WorldCoordinateHelper = new LV95WorldCoordinateHelper();
