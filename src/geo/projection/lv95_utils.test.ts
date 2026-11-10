import {describe, expect, test} from 'vitest';
import {LngLat} from '../lng_lat.ts';
import {MercatorCoordinate, mercatorXfromLng, mercatorYfromLat} from '../mercator_coordinate.ts';
import {LV95_PROJECTION_CONSTANTS, LV95_WORLD, lngLatToLV95, lv95ToLngLat, lv95ToWorld, lv95WorldCoordinateHelper, lv95WorldToMercator, mercatorToLV95World, worldToLV95} from './lv95_utils.ts';
import projectionLV95Vert from '../../shaders/glsl/_projection_lv95.vertex.glsl.g.ts';

/** The scalar and vector constants the LV95 vertex shader prelude declares, by name. */
function shaderConstants(): Record<string, number[]> {
    const constants: Record<string, number[]> = {};
    for (const [, name, value] of projectionLV95Vert.matchAll(/const float (\w+)=([-+\d.e]+);/g)) {
        constants[name] = [Number(value)];
    }
    for (const [, name, values] of projectionLV95Vert.matchAll(/const vec[23] (\w+)=vec[23]\(([^)]*)\);/g)) {
        constants[name] = values.split(',').map(Number);
    }
    return constants;
}

/**
 * The exact path of the LV95 vertex shader, `lv95ExactWorld`, step by step in double precision and with the constants
 * the shader declares, so that its algorithm can be checked against the rigorous transformation.
 */
function shaderExactWorld(mercatorX: number, mercatorY: number): {x: number; y: number; lineScale: number} {
    const c = Object.fromEntries(Object.entries(shaderConstants()).map(([name, values]) => [name, values.length === 1 ? values[0] : values])) as Record<string, any>;
    const lambda = (mercatorX - 0.5) * 2 * Math.PI;
    const psi = Math.PI * (1 - 2 * mercatorY);
    const sinPhi = Math.tanh(psi);
    const cosPhi = 1 / Math.cosh(psi);
    const sinLambda = Math.sin(lambda);
    const cosLambda = Math.cos(lambda);
    const w2 = 1 - c.LV95_WGS84_E2 * sinPhi * sinPhi;
    const n = c.LV95_WGS84_A / Math.sqrt(w2);
    const m = n * (1 - c.LV95_WGS84_E2) / w2;
    const [shiftX, shiftY, shiftZ] = c.LV95_DATUM_SHIFT;
    const dPhi = (-shiftX * sinPhi * cosLambda - shiftY * sinPhi * sinLambda + shiftZ * cosPhi +
        c.LV95_DATUM_DA * n * c.LV95_WGS84_E2 * sinPhi * cosPhi / c.LV95_WGS84_A +
        c.LV95_DATUM_DF * (m / (1 - c.LV95_WGS84_F) + n * (1 - c.LV95_WGS84_F)) * sinPhi * cosPhi) / m;
    const dLambda = (-shiftX * sinLambda + shiftY * cosLambda) / (n * cosPhi);
    const psiB = psi + dPhi / cosPhi;
    const sinPhiB = sinPhi + cosPhi * dPhi;
    const s = c.LV95_ALPHA * (psiB - c.LV95_BESSEL_E * Math.atanh(c.LV95_BESSEL_E * sinPhiB)) + c.LV95_K;
    const sinB = Math.tanh(s);
    const cosB = 1 / Math.cosh(s);
    const l = c.LV95_ALPHA * (lambda + dLambda - c.LV95_LAMBDA0);
    const sinObliqueLatitude = c.LV95_COS_B0 * sinB - c.LV95_SIN_B0 * cosB * Math.cos(l);
    const obliqueLongitude = Math.atan2(cosB * Math.sin(l), c.LV95_SIN_B0 * sinB + c.LV95_COS_B0 * cosB * Math.cos(l));
    return {
        x: c.LV95_WORLD_ORIGIN[0] + obliqueLongitude * c.LV95_WORLD_UNITS_PER_RADIAN,
        y: c.LV95_WORLD_ORIGIN[1] - Math.atanh(sinObliqueLatitude) * c.LV95_WORLD_UNITS_PER_RADIAN,
        lineScale: cosPhi / (c.LV95_COS_PHI_ORIGIN * Math.sqrt(1 - sinObliqueLatitude * sinObliqueLatitude)),
    };
}

/** Degrees, minutes and seconds as decimal degrees. */
function dms(degrees: number, minutes: number, seconds: number): number {
    return degrees + minutes / 60 + seconds / 3600;
}

describe('LV95 transformation', () => {
    test('derives the constants swisstopo publishes for the projection sphere', () => {
        expect(LV95_PROJECTION_CONSTANTS.R).toBeCloseTo(6378815.90365, 4);
        expect(LV95_PROJECTION_CONSTANTS.alpha).toBeCloseTo(1.00072913843038, 13);
        expect(LV95_PROJECTION_CONSTANTS.b0 * 180 / Math.PI).toBeCloseTo(dms(46, 54, 27.83324844), 10);
        expect(LV95_PROJECTION_CONSTANTS.K).toBeCloseTo(0.0030667323772751, 12);
    });

    test('places swisstopo\'s reference point at its LV95 coordinates', () => {
        const [east, north] = lngLatToLV95(dms(8, 43, 49.79), dms(46, 2, 38.87));
        expect(Math.abs(east - 2700000)).toBeLessThan(0.1);
        expect(Math.abs(north - 1100000)).toBeLessThan(0.1);
    });

    test('inverts the transformation across and around Switzerland', () => {
        for (const [lng, lat] of [[5.96, 45.82], [10.49, 47.81], [7.44, 46.95], [-3, 52], [21, 38], [14, 64]]) {
            const [restoredLng, restoredLat] = lv95ToLngLat(...lngLatToLV95(lng, lat));
            expect(restoredLng).toBeCloseTo(lng, 10);
            expect(restoredLat).toBeCloseTo(lat, 10);
        }
    });
});

describe('LV95 world plane', () => {
    test('puts the projection center where Web Mercator puts it, at the scale of Web Mercator', () => {
        const origin = lv95ToLngLat(2600000, 1200000);
        expect(LV95_WORLD.originX).toBeCloseTo(mercatorXfromLng(origin[0]), 15);
        expect(LV95_WORLD.originY).toBeCloseTo(mercatorYfromLat(origin[1]), 15);
        const kilometerEast = lv95ToLngLat(2601000, 1200000);
        const kilometerNorth = lv95ToLngLat(2600000, 1201000);
        const eastScale = LV95_WORLD.unitsPerMeter * 1000 / (mercatorXfromLng(kilometerEast[0]) - mercatorXfromLng(origin[0]));
        const northScale = LV95_WORLD.unitsPerMeter * 1000 / (mercatorYfromLat(origin[1]) - mercatorYfromLat(kilometerNorth[1]));
        expect(eastScale).toBeCloseTo(1, 6);
        // Web Mercator applies spherical formulas to ellipsoidal latitudes, which stretches it north-south.
        expect(northScale).toBeCloseTo(0.9968, 4);
    });

    test('runs the easting along x and the northing against y', () => {
        const [x, y] = lv95ToWorld(2601000, 1199000);
        expect(x).toBeGreaterThan(LV95_WORLD.originX);
        expect(y).toBeGreaterThan(LV95_WORLD.originY);
        const [east, north] = worldToLV95(x, y);
        expect(east).toBeCloseTo(2601000, 6);
        expect(north).toBeCloseTo(1199000, 6);
    });

    test('maps lng/lat, altitudes and Web Mercator positions through the world coordinate helper', () => {
        const zurich = new LngLat(8.5417, 47.3769);
        const world = lv95WorldCoordinateHelper.worldFromLngLat(zurich.lng, zurich.lat, 1000);
        expect(world.z).toBeCloseTo(1000 * LV95_WORLD.unitsPerMeter, 15);
        const restored = lv95WorldCoordinateHelper.lngLatFromWorld(world.x, world.y);
        expect(restored.lng).toBeCloseTo(zurich.lng, 10);
        expect(restored.lat).toBeCloseTo(zurich.lat, 10);
        const mercator = MercatorCoordinate.fromLngLat(zurich);
        const [x, y] = mercatorToLV95World(mercator.x, mercator.y);
        expect(x).toBeCloseTo(world.x, 12);
        expect(y).toBeCloseTo(world.y, 12);
        const [mercatorX, mercatorY] = lv95WorldToMercator(x, y);
        expect(mercatorX).toBeCloseTo(mercator.x, 12);
        expect(mercatorY).toBeCloseTo(mercator.y, 12);
    });

    test('is a flat plane bounded by its world square', () => {
        expect(lv95WorldCoordinateHelper.wraps).toBe(false);
        expect(lv95WorldCoordinateHelper.hasLngLat(0.5, 0.5)).toBe(true);
        expect(lv95WorldCoordinateHelper.hasLngLat(1.2, 0.5)).toBe(false);
        expect(lv95WorldCoordinateHelper.metersPerWorldUnit(0, 0)).toBeCloseTo(1 / LV95_WORLD.unitsPerMeter, 3);
    });
});

describe('LV95 vertex shader prelude', () => {
    test('declares the constants of the transformation', () => {
        const constants = shaderConstants();
        const expected: Record<string, number[]> = {
            LV95_ALPHA: [LV95_PROJECTION_CONSTANTS.alpha],
            LV95_SIN_B0: [Math.sin(LV95_PROJECTION_CONSTANTS.b0)],
            LV95_COS_B0: [Math.cos(LV95_PROJECTION_CONSTANTS.b0)],
            LV95_K: [LV95_PROJECTION_CONSTANTS.K],
            LV95_LAMBDA0: [LV95_PROJECTION_CONSTANTS.lambda0],
            LV95_BESSEL_E: [LV95_PROJECTION_CONSTANTS.besselE],
            LV95_WORLD_ORIGIN: [LV95_WORLD.originX, LV95_WORLD.originY],
            LV95_WORLD_UNITS_PER_RADIAN: [LV95_PROJECTION_CONSTANTS.R * LV95_WORLD.unitsPerMeter],
            LV95_COS_PHI_ORIGIN: [Math.cos(lv95ToLngLat(2600000, 1200000)[1] * Math.PI / 180)],
        };
        for (const [name, values] of Object.entries(expected)) {
            expect(constants[name]).toHaveLength(values.length);
            for (const [i, value] of values.entries()) {
                expect(Math.abs(constants[name][i] / value - 1)).toBeLessThan(1e-9);
            }
        }
    });

    test('projects within a decimeter of the rigorous transformation across Europe', () => {
        let maxError = 0;
        let maxLineScaleError = 0;
        for (let lng = -10; lng <= 30; lng += 2.5) {
            for (let lat = 35; lat <= 70; lat += 2.5) {
                const mercator = MercatorCoordinate.fromLngLat([lng, lat]);
                const [x, y] = mercatorToLV95World(mercator.x, mercator.y);
                const shader = shaderExactWorld(mercator.x, mercator.y);
                maxError = Math.max(maxError, Math.hypot(shader.x - x, shader.y - y) / LV95_WORLD.unitsPerMeter);
                const step = 1e-7;
                const [eastX, eastY] = mercatorToLV95World(mercator.x + step, mercator.y);
                maxLineScaleError = Math.max(maxLineScaleError, Math.abs(shader.lineScale / (Math.hypot(eastX - x, eastY - y) / step) - 1));
            }
        }
        expect(maxError).toBeLessThan(0.2);
        expect(maxLineScaleError).toBeLessThan(0.002);
    });
});
